import pygame
import torch
import numpy as np
from src.Environnement.envAgentvsHuman import PongEnvPlayer  # Utiliser PongEnvPlayer
from src.Agent.agent import PongAgent

def main():
    # Initialize environment and agent
    env = PongEnvPlayer()  # Instanciation correcte de la classe
    state_shape = (4, 5)  # Correspond à l'état original utilisé lors de l'entraînement
    n_actions = 3
    agent = PongAgent(state_shape, n_actions)
    
    # Load best model
    checkpoint = torch.load('models/pong_agent_best.pth', map_location=agent.device)
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0  # Disable exploration
    
    # Game loop
    running = True
    state = env.reset()
    state_stack = np.array([state for _ in range(4)])  # 4x5 = 20
    
    clock = pygame.time.Clock()
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Récupérer les touches enfoncées pour le joueur
        keys = pygame.key.get_pressed()
        player_action = 0  # Action par défaut : rester
        if keys[pygame.K_UP]:
            player_action = 1  # Monter
        elif keys[pygame.K_DOWN]:
            player_action = 2  # Descendre
        
        # Game logic
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(agent.device)
        action = agent.select_action(state_tensor)
        next_state, reward, done = env.step(action)
        
        state_stack = np.vstack([state_stack[1:], next_state])
        
        if done:
            state = env.reset()
            state_stack = np.array([state for _ in range(4)])
        
        # Render
        env._render_player()  # Utilisez la méthode appropriée pour le rendu
        clock.tick(60)  # Limit to 60 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()