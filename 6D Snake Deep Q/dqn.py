import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import *

class DQNNetwork(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for layer in FC_LAYERS:
            if len(layer) == 1:
                out_size = layer[0]
            else:
                out_size = layer[1]
            
            layers.append(nn.Linear(prev_size, out_size))
            layers.append(nn.ReLU())
            
            prev_size = out_size
        
        # Replace the last ReLU with the output layer
        layers[-1] = nn.Linear(prev_size, n_actions)
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.fc_layers(state)

class DQNAgent:
    def __init__(self, input_channels, n_actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE):       
        self.input_channels = input_channels
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Calculate input size based on the state shape
        self.input_size = input_channels
        
        self.q_network = DQNNetwork(self.input_size, n_actions).to(self.device)
        self.target_network = DQNNetwork(self.input_size, n_actions).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        
    def choose_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        #state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) # move to step

    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_models(self, path='dqn_model.pth'):
        torch.save(self.q_network.state_dict(), 'checkpoints/' + path)
    
    def load_models(self, path='dqn_model.pth'):
        self.q_network.load_state_dict(torch.load('checkpoints/' + path))
        self.target_network.load_state_dict(self.q_network.state_dict())
