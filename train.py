import numpy as np
import torch
from src.Environnement.env import PongEnv
from src.Agent.agent import PongAgent
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
import argparse
import time

print("CUDA disponible :", torch.cuda.is_available())

# Arguments pour activer le rendu
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Afficher la partie pendant l'entraînement")
args = parser.parse_args()
VISUALIZE = args.render

# Training parameters
EPISODES = 1250
SAVE_INTERVAL = 100
MODEL_DIR = "models"
LOG_DIR = "runs"
MOVING_AVG_WINDOW = 100

# Un seul environnement si on veut le rendu, sinon 8
NUM_ENVIRONMENTS = 1 if VISUALIZE else 8

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Si on veut le rendu, on désactive le mode training pour permettre l'affichage
envs = [PongEnv(training_mode=not VISUALIZE) for _ in range(NUM_ENVIRONMENTS)]
state_shape = (4, 5)  # Doit correspondre à l'environnement
n_actions = 3
agent = PongAgent(state_shape, n_actions)
writer = SummaryWriter(LOG_DIR) if not VISUALIZE else None

# Metrics tracking
best_score = float('-inf')
steps_done = 0
episode_rewards = deque(maxlen=MOVING_AVG_WINDOW)

try:
    for episode in tqdm(range(EPISODES), desc="Training", disable=VISUALIZE):
        states = [env.reset() for env in envs]
        state_stacks = [np.array([state for _ in range(4)]) for state in states]
        episode_reward = 0
        episode_length = 0
        episode_q_values = []
        dones = [False] * NUM_ENVIRONMENTS
        while not all(dones):
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                # Convertir state_stack en Tensor PyTorch
                state_tensor = torch.FloatTensor(state_stacks[i]).unsqueeze(0).to(agent.device)
                
                # Sélectionner l'action
                action = agent.select_action(state_tensor)
                
                next_state, reward, done = env.step(action)

                # Affichage du jeu si demandé (seulement pour le premier env)
                if VISUALIZE and i == 0:
                    env.render()
                    time.sleep(0.01)
                
                # Traquer les Q-values
                with torch.no_grad():
                    q_values = agent.policy_net(state_tensor)
                    episode_q_values.append(q_values.max().item())
                
                next_state_stack = np.vstack([state_stacks[i][1:], next_state])
                agent.store_transition(state_stacks[i], action, reward, next_state_stack, done)
                agent.train_step()
                agent.update_target_network(steps_done)
                
                state_stacks[i] = next_state_stack
                episode_reward += reward
                episode_length += 1
                steps_done += 1
                dones[i] = done

        # Traquer les métriques
        episode_rewards.append(episode_reward)
        moving_avg_reward = np.mean(episode_rewards)
        
        # Logger les métriques si pas en mode visualisation
        if writer:
            writer.add_scalar('Reward/Episode', episode_reward, episode)
            writer.add_scalar('Reward/Moving_Average', moving_avg_reward, episode)
            writer.add_scalar('Length/Episode', episode_length, episode)
            writer.add_scalar('Q_Value/Mean', np.mean(episode_q_values), episode)
            writer.add_scalar('Q_Value/Max', np.max(episode_q_values), episode)
            writer.add_scalar('Exploration/Epsilon', agent.epsilon, episode)

        print(f"Episode {episode}/{EPISODES} - Length: {episode_length} - "
              f"Reward: {episode_reward:.2f} - Avg Reward: {moving_avg_reward:.2f} - "
              f"Epsilon: {agent.epsilon:.3f}")
        
        if episode % SAVE_INTERVAL == 0:
            model_path = os.path.join(MODEL_DIR, f"pong_agent_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
            }, model_path)
            
            if episode_reward > best_score:
                best_score = episode_reward
                best_model_path = os.path.join(MODEL_DIR, "pong_agent_best.pth")
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                }, best_model_path)

finally:
    for env in envs:
        env.close()