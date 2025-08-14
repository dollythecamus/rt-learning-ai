# main.py
import pygame
from agent import Agent
from env import SCREEN_HEIGHT, SCREEN_WIDTH, FriendEnv


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