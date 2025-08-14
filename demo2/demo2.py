# demo_v2.py
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
import matplotlib.cm  # Import matplotlib's colormap functionality

# --- Hyperparameters ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PET_VISUAL_SIZE = 128 # The size of our NN visualization

# AI Hyperparameters
STATE_SIZE = 6  # [pet_x, pet_y, food_x, food_y, hunger, energy]
ACTION_SIZE = 4 # 0: MOVE, 1: EAT, 2: SLEEP, 3: IDLE
LEARNING_RATE = 0.01
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_LAYER_SIZE = 32 # The dimension of our square visualization

# --- The Neural Network (The "Meta-Controller") ---
class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, ACTION_SIZE)

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
        # Using a perceptually uniform colormap is great for data visualization
        self.colormap = matplotlib.colormaps.get_cmap('viridis')

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
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
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA * max_next_q_values)
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        return loss.item()

    # --- NEW METHOD FOR VISUALIZATION ---
    def get_appearance_surface(self):
        # 1. Get the weights from the second layer
        weights = self.policy_net.fc2.weight.detach().cpu().numpy()

        # 2. Normalize the weights to a 0-1 range for coloring
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())

        # 3. Apply the colormap to convert values to RGBA colors
        # The colormap returns values from 0-1, so we multiply by 255
        colored_weights = (self.colormap(norm_weights)[:, :, :3] * 255).astype(np.uint8)

        # 4. Create a Pygame surface from the colored pixel data
        # We need to create a surface from the raw numpy array of colors
        surface = pygame.surfarray.make_surface(colored_weights)
        
        # 5. Scale it up to a visible size
        return pygame.transform.scale(surface, (PET_VISUAL_SIZE, PET_VISUAL_SIZE))


# --- The Environment and Pet ---
class FriendEnv:
    def __init__(self):
        # Center pet in the screen
        self.pet_pos = [SCREEN_WIDTH // 2 - PET_VISUAL_SIZE // 2, SCREEN_HEIGHT // 2 - PET_VISUAL_SIZE // 2]
        self.food_pos = self._spawn_food()
        self.hunger = 100
        self.energy = 100
        self.is_sleeping = False
        self.sleep_timer = 0
        self.pet_size = PET_VISUAL_SIZE # Update pet size to match visual

    def _spawn_food(self):
        return [random.randint(0, SCREEN_WIDTH - 20), random.randint(0, SCREEN_HEIGHT - 20)]

    def get_state(self):
        return np.array([
            self.pet_pos[0] / SCREEN_WIDTH, self.pet_pos[1] / SCREEN_HEIGHT,
            self.food_pos[0] / SCREEN_WIDTH, self.food_pos[1] / SCREEN_HEIGHT,
            self.hunger / 100, self.energy / 100
        ])

    def step(self, action):
        reward = -0.1
        self.hunger = max(0, self.hunger - 0.1)
        self.energy = max(0, self.energy - 0.05)
        
        if self.is_sleeping:
            self.sleep_timer -= 1
            if self.sleep_timer <= 0: self.is_sleeping = False
            self.energy = min(100, self.energy + 1)
            return self.get_state(), reward, False

        if action == 0: # MOVE
            direction_x = self.food_pos[0] - self.pet_pos[0]
            direction_y = self.food_pos[1] - self.pet_pos[1]
            dist = math.hypot(direction_x, direction_y)
            if dist > 0:
                self.pet_pos[0] += direction_x / dist * 5
                self.pet_pos[1] += direction_y / dist * 5
                reward -= dist / 1000
        elif action == 1: # EAT
            dist_to_food = math.hypot(self.pet_pos[0] - self.food_pos[0], self.pet_pos[1] - self.food_pos[1])
            if dist_to_food < self.pet_size:
                if self.hunger < 80:
                    reward += 10
                    self.hunger = min(100, self.hunger + 50)
                    self.food_pos = self._spawn_food()
                else: reward -= 5
            else: reward -= 1
        elif action == 2: # SLEEP
            if self.energy < 50:
                reward += 10
                self.is_sleeping = True
                self.sleep_timer = 100
            else: reward -= 5
        if self.hunger < 20: reward -= 1
        if self.energy < 20: reward -= 1
        return self.get_state(), reward, False

    def draw(self, screen, agent): # <-- Pass the agent into the draw method
        pygame.draw.rect(screen, (255, 0, 0), (*self.food_pos, 15, 15))
        
        # --- MODIFIED DRAW LOGIC ---
        pet_surface = agent.get_appearance_surface()
        if self.is_sleeping:
            # Make it look desaturated when sleeping
            pet_surface.set_alpha(180) 
        
        screen.blit(pet_surface, self.pet_pos)
        
        pygame.draw.rect(screen, (0, 255, 0), (10, 10, self.hunger * 2, 15))
        pygame.draw.rect(screen, (255, 255, 0), (10, 30, self.energy * 2, 15))


# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Virtual Friend V2 - NN Visualization")
    clock = pygame.time.Clock()
    env = FriendEnv()
    agent = Agent()
    running = True
    font = pygame.font.Font(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = env.get_state()
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state)

        if env.is_sleeping:
            for _ in range(10): agent.replay()
        else:
            agent.replay()

        screen.fill((20, 20, 40))
        env.draw(screen, agent) # <-- Pass agent to draw method

        action_text = ["MOVE", "EAT", "SLEEP", "IDLE"][action]
        info_text = f"Action: {action_text} | Epsilon: {agent.epsilon:.2f}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 55))
        
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

if __name__ == '__main__':
    main()