
import argparse
import pygame
import torch
from snake_game import SnakeGame
from dqn import DQNAgent
from config import *
from renderer import Renderer
import random

def watch_simulation(model_path=None):
    renderer = Renderer()
    game = SnakeGame()
    state_shape = game.get_state().shape
    input_channels = state_shape[0]
    action_dim = len(game.direction)

    if model_path:
        agent = DQNAgent(input_channels=input_channels, n_actions=action_dim)
        agent.load_models()
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()
    else:
        agent = None

    running = True
    clock = pygame.time.Clock()

    while running:
        if game.done:
            game.reset()

        if agent:
            state = game.get_state()
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(agent.device)
            action = agent.choose_action(state)
        else:
            action = random.randint(0, action_dim - 1)

        game.step(action)

        renderer.render(game.snake, game.apples)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(10)  # Adjust the frame rate as needed

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch Snake Game Simulation")
    parser.add_argument("--model_path", type=str, help="Path to the saved model .pth file")
    args = parser.parse_args()

    watch_simulation(args.model_path)
