import numpy as np
import pygame

class PongEnv:
    def __init__(self, training_mode=False):
        # Constants
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        self.PADDLE_WIDTH = 20
        self.PADDLE_HEIGHT = 100
        self.BALL_SIZE = 15
        self.BALL_SPEED = 15
        self.BALL_SPEED_INCREASE = 1.1  # Speed multiplier
        self.MAX_BALL_SPEED = 25  # Maximum speed cap
        self.current_ball_speed = self.BALL_SPEED  # Track current speed
        self.PADDLE_SPEED = 20

        # Pygame setup
        pygame.init()
        self.training_mode = training_mode
        if not self.training_mode:
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        else:
            self.screen = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
       

        # Game state
        self.paddle_pos = np.array([50, self.WINDOW_HEIGHT/2])
        self.ball_pos = np.array([self.WINDOW_WIDTH/2, self.WINDOW_HEIGHT/2])
        self.ball_vel = np.array([self.BALL_SPEED, 0])
        
        # Score tracking
        self.score = 0
        self.done = False

    def render(self):
        # Affiche la fenêtre pygame si tu as déjà un affichage
        # Sinon, ajoute ici le code pour dessiner l'état du jeu
        if hasattr(self, 'screen'):
            import pygame
            pygame.display.flip()
            
    def reset(self):
        # Reset paddle and ball positions
        self.paddle_pos = np.array([50, self.WINDOW_HEIGHT/2])
        self.ball_pos = np.array([self.WINDOW_WIDTH/2, self.WINDOW_HEIGHT/2])
        
        # Random initial ball velocity
        self.current_ball_speed = self.BALL_SPEED  # Reset speed
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        self.ball_vel = np.array([np.cos(angle), np.sin(angle)]) * self.current_ball_speed
        
        self.score = 0
        self.done = False
        
        # Initialiser le compteur de touches
        self.hit_count = 0  # Ajout du compteur de touches

        return self._get_state()

    def step(self, action):
        # Action: 0 = stay, 1 = up, 2 = down
        reward = 0
        
        # Get positions before movement
        paddle_center = self.paddle_pos[1] + self.PADDLE_HEIGHT/2
        ball_center = self.ball_pos[1]
        
        # Calculate correct action based on positions
        correct_action = 0  # Default to stay
        if ball_center < paddle_center - self.PADDLE_HEIGHT/4:  # Ball is above paddle
            correct_action = 1  # Should move up
        elif ball_center > paddle_center + self.PADDLE_HEIGHT/4:  # Ball is below paddle
            correct_action = 2  # Should move down
        
        # Reward for correct action, penalty for wrong action
        if action == correct_action:
            reward += 0.5  # Reward for correct movement
        else:
            reward -= 0.2  # Small penalty for wrong movement
        
        # Move paddle
        if action == 1 and self.paddle_pos[1] > 0:
            self.paddle_pos[1] -= self.PADDLE_SPEED
        elif action == 2 and self.paddle_pos[1] < self.WINDOW_HEIGHT - self.PADDLE_HEIGHT:
            self.paddle_pos[1] += self.PADDLE_SPEED

        # Move ball
        self.ball_pos += self.ball_vel
        
        # Ball collisions
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.WINDOW_HEIGHT:
            self.ball_vel[1] *= -1

        if self.ball_pos[0] >= self.WINDOW_WIDTH:
            self.ball_vel[0] *= -1

        # Paddle collision (big reward)
        paddle_rect = pygame.Rect(self.paddle_pos[0], self.paddle_pos[1], 
                                self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_pos[0], self.ball_pos[1], 
                            self.BALL_SIZE, self.BALL_SIZE)
        
        if paddle_rect.colliderect(ball_rect):
            self.ball_vel[0] *= -1
            # Increase speed up to max
            self.current_ball_speed = min(self.current_ball_speed * self.BALL_SPEED_INCREASE, 
                                        self.MAX_BALL_SPEED)
            # Normalize direction vector and apply new speed
            direction = self.ball_vel / np.linalg.norm(self.ball_vel)
            self.ball_vel = direction * self.current_ball_speed
            reward += 10

            # Incrémenter le compteur de touches
            self.hit_count += 1
            if self.hit_count >= 5:  # Terminer l'épisode après 5 touches
                self.done = True

        # Ball out (big penalty)
        if self.ball_pos[0] <= 0:
            reward += -10  # Big penalty for missing ball
            self.done = True

        self._render()
        return self._get_state(), reward, self.done

    def _get_state(self):
        # Return normalized positions for the AI
        return np.array([
            self.paddle_pos[1] / self.WINDOW_HEIGHT,
            self.ball_pos[0] / self.WINDOW_WIDTH,
            self.ball_pos[1] / self.WINDOW_HEIGHT,
            self.ball_vel[0] / self.BALL_SPEED,
            self.ball_vel[1] / self.BALL_SPEED
        ])

    def _render(self):
            if self.training_mode:
                return
                
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Draw paddle
            pygame.draw.rect(self.screen, (255, 255, 255),
                            (self.paddle_pos[0], self.paddle_pos[1], 
                            self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
            
            # Draw ball
            pygame.draw.rect(self.screen, (255, 255, 255),
                            (self.ball_pos[0], self.ball_pos[1], 
                            self.BALL_SIZE, self.BALL_SIZE))
            
            if not self.training_mode:
                pygame.display.flip()
                self.clock.tick(60) 

    def close(self):
        pygame.quit()