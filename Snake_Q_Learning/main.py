
import sys
import pygame
import random
from snake_game import SnakeGame
from qlearning import initialize_q_table, update_q_value, get_q_value
from config import *
from snake_game_player import play_snake_game

def train_snake():
    # Initialize pygame
    pygame.init()

    # Initialize the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Snake Q-learning')

    # Initialize Q-table
    Q = initialize_q_table()

    max_length = 0
    moving_avg = []
    # Q-learning algorithm
    for episode in range(EPISODES):
        game = SnakeGame(screen)
        state = game.reset()
        done = False

        verbose = episode % RENDER_EVERY == 0

        while not done:
            if random.uniform(0, 1) < EPSILON:
                action = random.choice([0, 1, 2, 3])  # explore
            else:
                action = max(range(4), key=lambda a: get_q_value(Q, state, a))  # exploit

            next_state, reward, done = game.step(action)
            if verbose:
                print(f"\tReward: {reward:0.4f}")
            
            # Update Q-value
            update_q_value(Q, state, action, reward, next_state)

            state = next_state

            # Render the game
            if verbose:
                game.render()
                pygame.time.wait(25)
        
        snake_length = len(game.snake)
        moving_avg.insert(0, snake_length)
        moving_avg = moving_avg[:200]
        avg = sum(moving_avg)/len(moving_avg)
        if snake_length > max_length:
            max_length = snake_length
        print(f"Episode {episode + 1}/{EPISODES} completed, length={snake_length},avg={avg:0.2f} max={max_length}")

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play_snake_game()
    else:
        train_snake()
