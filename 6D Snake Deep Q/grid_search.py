
import argparse
import itertools
import json
from main import train_snake
from dqn import DQNAgent
from snake_game import SnakeGame
import random
import torch
import os

def grid_search(param_grid, num_episodes, num_runs):
    best_params = None
    best_avg_moving_avg = 0

    # Get all combinations of parameters
    param_combinations = list(itertools.product(*param_grid.values()))
    random.shuffle(param_combinations)

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")

        total_moving_avg = 0
        for run in range(num_runs):
            # Create a temporary game instance to get state and action dimensions
            temp_game = SnakeGame()
            state_shape = temp_game.get_state().shape
            input_channels = state_shape[0]
            action_dim = len(temp_game.direction)

            # Initialize DQN agent with current parameters
            agent = DQNAgent(
                input_channels=input_channels,
                n_actions=action_dim,
                alpha=param_dict['alpha'],
                gamma=param_dict['gamma'],
                epsilon=param_dict['epsilon'],
                epsilon_min=param_dict['epsilon_min'],
                epsilon_decay=param_dict['epsilon_decay'],
                batch_size=param_dict['batch_size'],
                memory_size=param_dict['memory_size']
            )

            # Train the agent
            _, _, moving_avg = train_snake(agent, num_episodes, render=False)
            total_moving_avg += moving_avg

        avg_moving_avg = total_moving_avg / num_runs
        print(f"Average moving avg: {avg_moving_avg}")

        if avg_moving_avg > best_avg_moving_avg:
            best_avg_moving_avg = avg_moving_avg
            best_params = param_dict
            
            # Write the new best parameters to a file
            write_best_params(best_params, best_avg_moving_avg)

    return best_params, best_avg_moving_avg

def write_best_params(params, avg_moving_avg):
    if not os.path.exists('grid_search_results'):
        os.makedirs('grid_search_results')
    
    filename = 'grid_search_results/best_params.json'
    data = {
        'parameters': params,
        'average_moving_avg': avg_moving_avg
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"New best parameters saved to '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for DQN Hyperparameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for each parameter combination")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each parameter combination")
    args = parser.parse_args()

    # Load parameter grid from JSON file
    with open(args.config, 'r') as f:
        param_grid = json.load(f)

    best_params, avg_moving_avg = grid_search(param_grid, args.episodes, args.runs)

    print(f"\nBest parameters: {best_params}")
    print(f"Best average moving avg: {avg_moving_avg}")

    # Final save of best parameters (redundant, but keeps the original behavior)
    write_best_params(best_params, avg_moving_avg)
