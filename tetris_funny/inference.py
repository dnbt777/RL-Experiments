
import numpy as np
import pickle
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
from config import *

def run_inference():
    env = TetrisEnv()
    state_size = BOARD_WIDTH * BOARD_HEIGHT
    action_size = 4  # Left, Right, Rotate, Drop
    agent = DQNAgent(state_size, action_size)
    agent.load("tetris_dqn_final.h5")
    agent.epsilon = 0  # No exploration during inference

    gameplay_data = []

    for game in range(NUM_GAMES):
        state = env.reset()
        game_states = []
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            game_states.append(state.reshape(BOARD_HEIGHT, BOARD_WIDTH))
            state = next_state
            total_reward += reward

            if done:
                break

        gameplay_data.append({
            'game_number': game + 1,
            'score': total_reward,
            'states': game_states
        })

        print(f"Game {game + 1}: Score = {total_reward}")

    with open(RECORD_FILE, 'wb') as f:
        pickle.dump(gameplay_data, f)

if __name__ == "__main__":
    run_inference()
