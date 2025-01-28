import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        input_dim = input_shape[0] * input_shape[1]  # 4 * 5 = 20
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten correctly while preserving batch dimension
        return self.fc(x)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PongAgent:
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Agent initialisé sur le dispositif : {self.device}")
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.memory = deque(maxlen=5000)
        self.priorities = deque(maxlen=5000)  # Add priorities
        self.alpha = 0.6  # Priority level
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.target_update = 1000
        self.frame_stack = deque(maxlen=4)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities, default=1.0))  # Initial priority

    def sample_experiences(self):
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]
        return experiences, indices

    def stack_frames(self, frame):
        self.frame_stack.append(frame / 255.0)  # Normalize pixels
        if len(self.frame_stack) < 4:
            for _ in range(4 - len(self.frame_stack)):
                self.frame_stack.append(frame / 255.0)
        return torch.FloatTensor(np.array(self.frame_stack)).unsqueeze(0)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            # State est déjà un tensor sur le bon appareil
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences, indices = self.sample_experiences()
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Double DQN: select actions with policy network
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for next_actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        for idx in indices:
            self.priorities[idx] = loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self, steps_done):
        if steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())