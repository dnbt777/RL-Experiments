
import pygame
import sys
from snake_game import SnakeGame
from config import *

def play_snake_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Snake Game - Player')
    clock = pygame.time.Clock()

    game = SnakeGame(screen)
    game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                else:
                    continue

                _, reward, done = game.step(action)
                if done:
                    print(f"Game Over! Score: {len(game.snake)}")
                    pygame.quit()
                    return

        game.render()
        clock.tick(10)  # Control game speed

if __name__ == "__main__":
    play_snake_game()
