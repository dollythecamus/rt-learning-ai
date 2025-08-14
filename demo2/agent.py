# agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# AI Hyperparameters
STATE_SIZE = 6  # [pet_x, pet_y, food_x, food_y, hunger, energy]
ACTION_SIZE = 4 # 0: MOVE, 1: EAT, 2: SLEEP, 3: IDLE
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for future rewards
EPSILON_START = 1.0 # Exploration rate
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64

# --- The Neural Network (The "Meta-Controller") ---
class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, ACTION_SIZE)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- The Agent (The "Brain") ---
class Agent:
    def __init__(self):
        self.policy_net = DQNetwork()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        # Epsilon-Greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)  # Explore
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item() # Exploit

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self):
        # Don't train until we have enough memories
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0

        # Sample a random batch of memories
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))

        # Get current Q values for chosen actions
        current_q_values = self.policy_net(states).gather(1, actions)

        # Get max Q values for next states
        with torch.no_grad():
            max_next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
        
        # Calculate target Q values using the Bellman equation
        target_q_values = rewards + (GAMMA * max_next_q_values)

        # Calculate loss and perform backpropagation
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon to reduce exploration over time
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
        return loss.item()
