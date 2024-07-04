
import pygame
import sys
from snake_game import SnakeGame
from renderer import Renderer
from config import *

def play_game():
    pygame.init()
    renderer = Renderer()
    game = SnakeGame()
    clock = pygame.time.Clock()

    # Map keys to directions
    key_to_direction = {
        pygame.K_s: (0, -1, 0, 0, 0, 0),  # Up (switched)
        pygame.K_w: (0, 1, 0, 0, 0, 0),   # Down (switched)
        pygame.K_a: (-1, 0, 0, 0, 0, 0),  # Left
        pygame.K_d: (1, 0, 0, 0, 0, 0),   # Right
        pygame.K_q: (0, 0, -1, 0, 0, 0),  # Back
        pygame.K_e: (0, 0, 1, 0, 0, 0),   # Forward
        pygame.K_j: (0, 0, 0, -1, 0, 0),  # W-
        pygame.K_l: (0, 0, 0, 1, 0, 0),   # W+
        pygame.K_i: (0, 0, 0, 0, 1, 0),   # V+
        pygame.K_k: (0, 0, 0, 0, -1, 0),  # V-
        pygame.K_u: (0, 0, 0, 0, 0, -1),  # U-
        pygame.K_o: (0, 0, 0, 0, 0, 1),   # U+
    }

    running = True
    while running:
        # Print current state and reward before each action
        current_state = game.get_state()
        print(f"Current State: {current_state}")
        print(f"Current Reward: {game.net_reward}")
        
        # Print current position of the apple and snake head
        print(f"Apple position: {game.target_apple}")
        print(f"Snake head position: {game.snake[0]}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key in key_to_direction:
                    new_direction = key_to_direction[event.key]
                    action = BASIS_DIRECTIONS.index(new_direction)
                elif event.unicode.isdigit():
                    action = int(event.unicode)
                    if action < 0 or action >= len(BASIS_DIRECTIONS):
                        action = None

                if action is not None:
                    _, reward, done = game.step(action)
                    print(f"Action taken: {action}")
                    print(f"Reward: {reward}")
                    if done:
                        print(f"Game Over! Score: {len(game.snake)}")
                        game.reset()

        renderer.render(game.snake, game.apples)
        clock.tick(10)  # Adjust the game speed here

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_game()
