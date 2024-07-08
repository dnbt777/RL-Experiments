
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import *

class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_shape, action_space).to(self.device)
        self.target_model = DQN(state_shape, action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.total_steps = 0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return {
                'rotate': random.choice([-1, 0, 1]),
                'move': random.choice([-1, 0, 1]),
                'color': random.choice([0, 1])
            }
        else:
            state_tensor = self.preprocess_state(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_idx = q_values.argmax().item()
            return self.idx_to_action(action_idx)
        

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat([self.preprocess_state(s) for s in states])
        next_states = torch.cat([self.preprocess_state(s) for s in next_states])
        actions = torch.tensor([self.action_to_idx(a) for a in actions], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.total_steps += 1
        if self.total_steps % SAVE_MODEL_STEPS == 0:
            self.save_model_checkpoint()

    def save_model_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, f'model_checkpoints/model_step_{self.total_steps}.pth')
        print(f"Model checkpoint saved at step {self.total_steps}")
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def preprocess_state(self, state):
        grid = torch.tensor(state['grid'], dtype=torch.float32).unsqueeze(0)
        canvas = torch.tensor(state['canvas'], dtype=torch.float32).unsqueeze(0)
        
        current_piece = torch.zeros((1, self.state_shape[1], self.state_shape[2]), dtype=torch.float32)
        if state['current_piece']:
            shape = state['current_piece']['shape']
            x, y = state['current_piece']['x'], state['current_piece']['y']
            color = state['current_piece']['color']
            for i in range(len(shape)):
                for j in range(len(shape[0])):
                    if shape[i][j]:
                        current_piece[0, y+i, x+j] = color
        
        next_piece = torch.zeros((1, self.state_shape[1], self.state_shape[2]), dtype=torch.float32)
        shape = state['next_piece']
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j]:
                    next_piece[0, i, j] = 1
        
        state_tensor = torch.cat([grid, canvas, current_piece, next_piece], dim=0).unsqueeze(0)
        return state_tensor.to(self.device)
    
    def action_to_idx(self, action):
        rotate = action['rotate'] + 1
        move = action['move'] + 1
        color = action['color']
        return rotate * 6 + move * 2 + color
    
    def idx_to_action(self, idx):
        rotate = (idx // 6) - 1
        move = ((idx % 6) // 2) - 1
        color = idx % 2
        return {
            'rotate': rotate,
            'move': move,
            'color': color
        }

class DQN(nn.Module):
    def __init__(self, state_shape, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        conv_out_size = self._get_conv_out(state_shape)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_space)
        
    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_dqn_agent(grid_width, grid_height):
    state_shape = (4, grid_height, grid_width)  # 4 channels: grid, canvas, current_piece, next_piece
    action_space = 18  # 3 rotate * 3 move * 2 color
    return DQNAgent(state_shape, action_space)
