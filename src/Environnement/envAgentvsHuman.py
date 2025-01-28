import pygame
import numpy as np
from src.Environnement.env import PongEnv

class PongEnvPlayer(PongEnv):
    def __init__(self):
        super().__init__()
        # Override speeds for player vs agent mode
        self.PADDLE_SPEED = 20  # Increased from 5
        self.BALL_SPEED = 15  # Increased from 7
        self.BALL_SPEED_INCREASE = 1.2  # Bigger speed increase
        self.MAX_BALL_SPEED = 20  # Higher max speed
        self.current_ball_speed = self.BALL_SPEED
        
        self.player_paddle_pos = np.array([self.WINDOW_WIDTH - 70, self.WINDOW_HEIGHT/2])
        self.player_score = 0
        self.agent_score = 0
        
    def reset(self):
        state = super().reset()
        self.current_ball_speed = self.BALL_SPEED  # Reset speed
        self.player_paddle_pos = np.array([self.WINDOW_WIDTH - 70, self.WINDOW_HEIGHT/2])
        self.player_score = 0
        self.agent_score = 0
        return self._get_state()

    def step(self, action):
        # Handle player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.player_paddle_pos[1] > 0:
            self.player_paddle_pos[1] -= self.PADDLE_SPEED
        if keys[pygame.K_DOWN] and self.player_paddle_pos[1] < self.WINDOW_HEIGHT - self.PADDLE_HEIGHT:
            self.player_paddle_pos[1] += self.PADDLE_SPEED

        # Move agent paddle
        if action == 1 and self.paddle_pos[1] > 0:
            self.paddle_pos[1] -= self.PADDLE_SPEED
        elif action == 2 and self.paddle_pos[1] < self.WINDOW_HEIGHT - self.PADDLE_HEIGHT:
            self.paddle_pos[1] += self.PADDLE_SPEED

        # Determine correct action
        paddle_center = self.paddle_pos[1] + self.PADDLE_HEIGHT/2
        ball_center = self.ball_pos[1]
        correct_action = 0
        if ball_center < paddle_center - self.PADDLE_HEIGHT/4:
            correct_action = 1
        elif ball_center > paddle_center + self.PADDLE_HEIGHT/4:
            correct_action = 2

        # Reward for correct/wrong action
        reward = 0
        if action == correct_action:
            reward += 0.5
        else:
            reward -= 0.2

        # Always move the ball here
        self.ball_pos += self.ball_vel
      

            # Wall collisions
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.WINDOW_HEIGHT:
            self.ball_vel[1] *= -1

        # Paddle collisions
        agent_paddle = pygame.Rect(self.paddle_pos[0], self.paddle_pos[1], 
                                self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        player_paddle = pygame.Rect(self.player_paddle_pos[0], self.player_paddle_pos[1], 
                                self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_pos[0], self.ball_pos[1], 
                            self.BALL_SIZE, self.BALL_SIZE)

        reward = 0
        # Agent paddle hit
        if agent_paddle.colliderect(ball_rect):
            reward += 10 
            # Increase speed
            self.current_ball_speed = min(self.current_ball_speed * self.BALL_SPEED_INCREASE, 
                                        self.MAX_BALL_SPEED)
            # Add angle variation based on where ball hits paddle
            relative_intersect_y = (self.paddle_pos[1] + self.PADDLE_HEIGHT/2) - self.ball_pos[1]
            normalized_intersect = relative_intersect_y / (self.PADDLE_HEIGHT/2)
            bounce_angle = normalized_intersect * np.pi/3  # 60 degrees max angle
            
            self.ball_vel = np.array([
                self.current_ball_speed * np.cos(bounce_angle),
                -self.current_ball_speed * np.sin(bounce_angle)
            ])
            reward = 10

      
        # Player paddle hit
        if player_paddle.colliderect(ball_rect):
            self.current_ball_speed = min(self.current_ball_speed * self.BALL_SPEED_INCREASE, 
                                        self.MAX_BALL_SPEED)
            relative_intersect_y = (self.player_paddle_pos[1] + self.PADDLE_HEIGHT/2) - self.ball_pos[1]
            normalized_intersect = relative_intersect_y / (self.PADDLE_HEIGHT/2)
            bounce_angle = normalized_intersect * np.pi/3
            
            self.ball_vel = np.array([
                -self.current_ball_speed * np.cos(bounce_angle),
                -self.current_ball_speed * np.sin(bounce_angle)
            ])

        # Goals
           # Goals with same rewards as training
        if self.ball_pos[0] <= 0:  # Player scores
            self.player_score += 1
            reward = -10  # Same penalty as training
            self.done = True
        elif self.ball_pos[0] >= self.WINDOW_WIDTH:  # Agent scores
            self.agent_score += 1
            reward = 10  # Same reward as training
            self.done = True

        return self._get_state(), reward, self.done

    def _render_player(self):
        self.screen.fill((0, 0, 0))
        
        # Draw paddles
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.paddle_pos[0], self.paddle_pos[1], 
                         self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.player_paddle_pos[0], self.player_paddle_pos[1], 
                         self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.ball_pos[0], self.ball_pos[1], 
                         self.BALL_SIZE, self.BALL_SIZE))
        
        # Draw scores
        font = pygame.font.Font(None, 36)
        agent_text = font.render(str(self.agent_score), True, (255, 255, 255))
        player_text = font.render(str(self.player_score), True, (255, 255, 255))
        self.screen.blit(agent_text, (self.WINDOW_WIDTH//4, 20))
        self.screen.blit(player_text, (3*self.WINDOW_WIDTH//4, 20))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    

    def _get_state(self):
        # Return normalized positions for the AI (exclude player's paddle position)
        return np.array([
            self.paddle_pos[1] / self.WINDOW_HEIGHT,
            self.ball_pos[0] / self.WINDOW_WIDTH,
            self.ball_pos[1] / self.WINDOW_HEIGHT,
            self.ball_vel[0] / self.BALL_SPEED,
            self.ball_vel[1] / self.BALL_SPEED
        ])