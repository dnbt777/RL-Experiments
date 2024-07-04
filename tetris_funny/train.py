
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
from config import *
import numpy as np

def train():
    env = TetrisEnv()
    state_size = BOARD_WIDTH * BOARD_HEIGHT
    action_size = 4  # Left, Right, Rotate, Drop
    agent = DQNAgent(state_size, action_size)

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.replay(BATCH_SIZE)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if episode % 1000 == 0:
            agent.save(f"tetris_dqn_{episode}.h5")

    agent.save("tetris_dqn_final.h5")

if __name__ == "__main__":
    train()
