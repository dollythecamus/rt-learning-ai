# main.py
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math

# --- Hyperparameters ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
PET_SIZE = 20
FOOD_SIZE = 15

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

# --- The Environment and Pet ---
class FriendEnv:
    def __init__(self):
        self.pet_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        self.food_pos = self._spawn_food()
        self.hunger = 100 # 0 is very hungry, 100 is full
        self.energy = 100 # 0 is very tired, 100 is rested
        self.is_sleeping = False
        self.sleep_timer = 0
        
    def _spawn_food(self):
        return [random.randint(0, SCREEN_WIDTH - FOOD_SIZE), random.randint(0, SCREEN_HEIGHT - FOOD_SIZE)]

    def get_state(self):
        # Normalize state values to be between 0 and 1 for the NN
        return np.array([
            self.pet_pos[0] / SCREEN_WIDTH,
            self.pet_pos[1] / SCREEN_HEIGHT,
            self.food_pos[0] / SCREEN_WIDTH,
            self.food_pos[1] / SCREEN_HEIGHT,
            self.hunger / 100,
            self.energy / 100
        ])

    def step(self, action):
        # Action: 0: MOVE, 1: EAT, 2: SLEEP, 3: IDLE
        reward = -0.1 # Small penalty for existing, encourages efficiency
        
        # Update needs over time
        self.hunger = max(0, self.hunger - 0.1)
        self.energy = max(0, self.energy - 0.05)
        
        if self.is_sleeping:
            self.sleep_timer -= 1
            if self.sleep_timer <= 0:
                self.is_sleeping = False
            self.energy = min(100, self.energy + 1)
            return self.get_state(), reward, False

        # --- Execute Action ---
        if action == 0: # MOVE
            # This is where a specialized "movement module" would go.
            # For this demo, we use a simple greedy logic.
            direction_x = self.food_pos[0] - self.pet_pos[0]
            direction_y = self.food_pos[1] - self.pet_pos[1]
            dist = math.hypot(direction_x, direction_y)
            
            if dist > 0:
                self.pet_pos[0] += direction_x / dist * 5
                self.pet_pos[1] += direction_y / dist * 5
                reward -= dist / 1000 # Penalize distance to encourage getting closer
        
        elif action == 1: # EAT
            dist_to_food = math.hypot(self.pet_pos[0] - self.food_pos[0], self.pet_pos[1] - self.food_pos[1])
            if dist_to_food < PET_SIZE:
                if self.hunger < 80:
                    reward += 10 # Big reward for eating when hungry
                    self.hunger = min(100, self.hunger + 50)
                    self.food_pos = self._spawn_food()
                else:
                    reward -= 5 # Penalty for eating when not hungry
            else:
                reward -= 1 # Penalty for trying to eat far from food

        elif action == 2: # SLEEP
            if self.energy < 50:
                reward += 10 # Big reward for sleeping when tired
                self.is_sleeping = True
                self.sleep_timer = 100 # Sleep for 100 frames
            else:
                reward -= 5 # Penalty for sleeping when not tired
        
        # Action 3 (IDLE) does nothing, just incurs the small time penalty

        # Add penalties for low needs
        if self.hunger < 20: reward -= 1
        if self.energy < 20: reward -= 1
            
        return self.get_state(), reward, False # `done` is always False for continuous learning

    def draw(self, screen):
        # Draw Food
        pygame.draw.rect(screen, (255, 0, 0), (*self.food_pos, FOOD_SIZE, FOOD_SIZE))
        
        # Draw Pet
        pet_color = (0, 150, 255) # Blue
        if self.is_sleeping:
            pet_color = (150, 150, 150) # Gray
        pygame.draw.circle(screen, pet_color, self.pet_pos, PET_SIZE)
        
        # Draw status bars
        pygame.draw.rect(screen, (0, 255, 0), (10, 10, self.hunger * 2, 15)) # Hunger
        pygame.draw.rect(screen, (255, 255, 0), (10, 30, self.energy * 2, 15)) # Energy


# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Virtual Friend Demo")
    clock = pygame.time.Clock()

    env = FriendEnv()
    agent = Agent()
    running = True
    total_reward = 0
    loss = 0

    font = pygame.font.Font(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 1. Get current state
        state = env.get_state()
        
        # 2. Agent chooses an action
        action = agent.choose_action(state)

        # 3. Environment updates based on action
        next_state, reward, done = env.step(action)
        total_reward += reward

        # 4. Agent remembers this experience
        agent.remember(state, action, reward, next_state)

        # 5. Agent learns from memories
        if env.is_sleeping:
            # During sleep, learn much more intensely
            for _ in range(10): 
                loss = agent.replay()
        else:
            # Regular learning
            loss = agent.replay()

        # --- Drawing ---
        screen.fill((20, 20, 40)) # Dark blue background
        env.draw(screen)

        # Display info
        action_text = ["MOVE", "EAT", "SLEEP", "IDLE"][action]
        info_text = f"Action: {action_text} | Epsilon: {agent.epsilon:.2f} | Loss: {loss:.4f}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 55))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    pygame.quit()

if __name__ == '__main__':
    main()