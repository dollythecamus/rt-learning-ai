import random
import math
import numpy as np
import pygame

# --- Hyperparameters ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
PET_SIZE = 20
FOOD_SIZE = 15

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
