
import sys
import pygame
import random
import os
import time
import pickle
import torch
from snake_game import SnakeGame
from dqn import DQNAgent
from config import *
from renderer import Renderer
from utils import *

def save_max_run(game, episode, timestamp):
    if not os.path.exists('./run_saves'):
        os.makedirs('./run_saves')
    
    filename = f'./run_saves/maxrun{timestamp}.save'
    data = {
        'snake': game.snake,
        'apples': game.apples,
        'direction': game.direction,
        'episode': episode,
        'length': len(game.snake)
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def train_snake(agent=None, num_episodes=EPISODES, render=False, load_model=None):
    renderer = Renderer() if render else None

    temp_game = SnakeGame()
    state_shape = temp_game.get_state().shape
    input_dim = state_shape[0]
    action_dim = len(BASIS_DIRECTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if agent is None:
        agent = DQNAgent(input_channels=input_dim, n_actions=action_dim)

    if load_model:
        agent.load_models(load_model)
        print(f"Loaded model from {load_model}")

    max_length = 0
    moving_avg = []
    total_steps = 0
    timestamp = int(time.time())

    last_episode_time = time.time()
    next_save_steps = MODEL_SAVE_INTERVAL

    step_counter = 0

    for episode in range(num_episodes):
        game = SnakeGame()
        state = game.get_state()
        done = False

        verbose = render and episode % RENDER_EVERY == 0
        if verbose:
            renderer.reset_param_space()

        episode_return = 0
        steps = 0
        
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.choose_action(state_tensor)
            next_state, reward, done = game.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            
            agent.remember(state_tensor, action, reward, next_state_tensor, done)
            
            episode_return += reward
            steps += 1
            total_steps += 1
            step_counter += 1

            if total_steps % 4 == 0:  # Reduce learning frequency
                agent.learn()

            if total_steps % 100 == 0:
                agent.update_target_network()

            state = next_state

            if verbose:
                snake, food = game.get_render_data()
                renderer.render(snake, food)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return max_length

                pygame.time.wait(MS_PER_FRAME)

        # Save model every MODEL_SAVE_INTERVAL steps
        if total_steps > next_save_steps:
            agent.save_models(f"model_steps_{total_steps}.pth")
            print(f"Model saved at step {total_steps}")
            next_save_steps += MODEL_SAVE_INTERVAL
        
        snake_length = len(game.snake)
        moving_avg.append(snake_length)
        if len(moving_avg) > 200:
            moving_avg.pop(0)
        avg = sum(moving_avg) / len(moving_avg)
        
        if snake_length > max_length:
            max_length = snake_length
        
        if episode % PRINT_EVERY == 0:
            current_episode_time = time.time()
            time_passed = (current_episode_time - last_episode_time)
            last_episode_time = current_episode_time

            print(f"Episode {episode + 1}/{num_episodes}, e/s={PRINT_EVERY/time_passed:5.2f} steps/s={step_counter/time_passed:5.2f}")
            print(f"Steps={steps}, ep_return={episode_return:6.2f} length={snake_length:3f}, avg={avg:3.2f} max={max_length:3f}")
            print(f"Steps: {100*game.steps_towards_apple/notzero(game.steps_towards_apple+game.steps_away_from_apple):3.0f}% Towards: {game.steps_towards_apple:3f} Away:{game.steps_away_from_apple:3f}")
            print(f"Death: {game.death_type}, Last direction: {game.direction} {BASIS_DIRECTIONS[action]}")
            print(f"Epsilon: {agent.epsilon:1.4f}")
            print(state)
            print(state_tensor)
            print(action)
            print(game.snake[0])
            print()
            
            step_counter = 0
            

    if render:
        agent.save_models()
        pygame.quit()

    return max_length, avg, episode_return

def replay_max_run(timestamp):
    filename = f'./run_saves/maxrun{timestamp}.save'
    if not os.path.exists(filename):
        print(f"No saved run found for timestamp {timestamp}")
        return

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    renderer = Renderer()
    game = SnakeGame()
    game.snake = data['snake']
    game.apples = data['apples']
    game.direction = data['direction']

    print(f"Replaying max run from episode {data['episode']} with length {data['length']}")

    running = True
    while running:
        renderer.render(game.snake, game.apples)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.time.wait(100)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            print('Play mode is disabled')
        elif sys.argv[1] == "replay":
            if len(sys.argv) > 2:
                replay_max_run(sys.argv[2])
            else:
                print("Please provide a timestamp for replay")
        elif sys.argv[1] == "load":
            if len(sys.argv) > 2:
                train_snake(load_model=sys.argv[2])
            else:
                print("Please provide a model path to load")
        else:
            print("Invalid argument. Use 'play', 'replay <timestamp>', or 'load <model_path>'")
    else:
        train_snake()
