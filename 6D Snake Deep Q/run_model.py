
import argparse
import torch
import pygame
from snake_game import SnakeGame
from dqn import DQNAgent
from config import *
from renderer import Renderer
import time
import numpy as np

def run_model(model_path, render=False, num_episodes=100):
    renderer = Renderer() if render else None
    game = SnakeGame()
    state_shape = game.get_state().shape
    input_channels = state_shape[0]
    action_dim = len(BASIS_DIRECTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(input_channels=input_channels, n_actions=action_dim, epsilon=0, epsilon_min=0)
    agent.load_models(model_path)
    agent.q_network.eval()
    print(agent.epsilon, agent.epsilon_min, "asd09as09jas")
    episode_lengths = []
    episode_rewards = []
    episode_apples = []
    total_steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        state = game.reset()
        episode_reward = 0
        episode_steps = 0

        while not game.done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.choose_action(state_tensor)
            next_state, reward, done = game.step(action)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if render:
                snake, food = game.get_render_data()
                renderer.render(snake, food)
                pygame.time.wait(50)  # Adjust speed of rendering

            state = next_state

        episode_lengths.append(episode_steps)
        episode_rewards.append(episode_reward)
        episode_apples.append(game.apples_collected)

        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Length: {episode_steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Apples collected: {game.apples_collected}")
        print(f"  Steps towards apple: {game.steps_towards_apple}")
        print(f"  Steps away from apple: {game.steps_away_from_apple}")
        print(f"  Death type: {game.death_type}")
        print()

    end_time = time.time()
    total_time = end_time - start_time

    print("\nOverall Statistics:")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average apples collected: {np.mean(episode_apples):.2f} ± {np.std(episode_apples):.2f}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Steps per second: {total_steps / total_time:.2f}")

    if render:
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained Snake model")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--render", action="store_true", help="Render the game while running")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    args = parser.parse_args()

    run_model(args.model_path, args.render, args.episodes)
