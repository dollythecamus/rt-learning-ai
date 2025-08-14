
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque #dunno what this is

import random

# constants relating to neural network size
STATE_SIZE = 1 # inputs
ACTION_SIZE = 1 # outputs
HIDDEN_LAYER_SIZE = 16 # layer

# constants relating to learning
LEARNING_RATE = 1
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64

GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_DECAY = 1.0
EPSILON_END = 0.1

# neural network
class yuriNN(nn.Module):
    def __init__(self):
        super(yuriNN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) # can add as many of these layers as i wish, i suppose
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) # can add as many of these layers as i wish
        self.fc4 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) # can add as many of these layers as i wish
        self.fc5 = nn.Linear(HIDDEN_LAYER_SIZE, ACTION_SIZE)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x)) 
        x = torch.relu(self.fc4(x)) #adding more layers means also forwarding the layers, dummy
        return self.fc5(x)
    

# part 1: learning algorythm
class Agent:
    def __init__(self):
        self.yuriNN = yuriNN()
        self.optimizer = optim.Adam(self.yuriNN.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.yuriNN(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        current_q_values = self.yuriNN(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.yuriNN(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA * max_next_q_values)
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        return loss.item()
