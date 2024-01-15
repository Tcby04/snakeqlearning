import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle


# Use the Tkinter backend for matplotlib
matplotlib.use('TkAgg')

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset_game()

    def reset_game(self):
        self.snake = [(self.width//2, self.height//2)]
        self.score = 0
        self.food = None
        self.place_food()
        self.game_over = False

    def place_food(self):
        while self.food is None or self.food in self.snake:
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))

    def play_step(self, action):
        assert action in ['U', 'D', 'L', 'R'], "Invalid action."

        # Move snake
        x, y = self.snake[0]
        if action == 'U': y -= 1
        if action == 'D': y += 1
        if action == 'L': x -= 1
        if action == 'R': x += 1

        self.snake.insert(0, (x, y))

        # Check if game over
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake[1:]:
            self.game_over = True
            return self.score, self.game_over

        # Check if got food
        if (x, y) == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        return self.score, self.game_over

class SnakeGameVisual(SnakeGame):
    def __init__(self, width=10, height=10):
        super().__init__(width, height)
        plt.ion()  # Turn on the interactive mode

    def draw_board(self, episode=None, move_counter=None):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.clear()
        self.ax.set_xlim(-1, self.width)
        self.ax.set_ylim(-1, self.height)
        self.ax.grid(False)
        self.ax.set_xticks(range(self.width))
        self.ax.set_yticks(range(self.height))

        title = f"Score: {self.score} | Moves Left: {self.moves_left}"
        if episode is not None:
            title += f" | Gen: {episode}"
        self.ax.set_title(title)


        # Draw snake
        for x, y in self.snake:
            snake_rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, edgecolor='black', facecolor='green')
            self.ax.add_patch(snake_rect)

        # Draw food
        fx, fy = self.food
        food_rect = patches.Rectangle((fx-0.5, fy-0.5), 1, 1, edgecolor='black', facecolor='red')
        self.ax.add_patch(food_rect)

        plt.draw()
        plt.pause(0.1)  # Pause for a short period to visualize the move

        

class SnakeGameAI(SnakeGameVisual):
    def __init__(self, width=10, height=10, epsilon=100, epsilon_decay=0.5, alpha=.01, gamma=0.5):
        self.MAX_MOVES_PER_GAME = 100  # Initialize MAX_MOVES_PER_GAME first
        super().__init__(width, height)
        self.epsilon = epsilon        # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.alpha = alpha            # Learning rate
        self.gamma = gamma            # Discount factor
        self.q_table = {}             # Initialize Q-table
        self.scores = []              # To store scores of each episode


    def board_coverage(self):
        # Calculate how much of the board is covered by the snake
        return len(self.snake) / (self.width * self.height)

    def get_state(self):

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state = (
        # Relative food position
        food_x < head_x,  # Food left
        food_x > head_x,  # Food right
        food_y < head_y,  # Food up
        food_y > head_y,  # Food down
        # Danger straight ahead, left, or right
        self.is_danger('U'),
        self.is_danger('L'),
        self.is_danger('R'),
        self.is_danger('D')
                )
        return state
    
    def is_danger(self, direction):
        """
        Check if moving in a specific direction would result in hitting a wall or itself.
        """
        x, y = self.snake[0]
        if direction == 'U': y -= 1
        if direction == 'D': y += 1
        if direction == 'L': x -= 1
        if direction == 'R': x += 1
        return x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake

    def get_action(self, state):
        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            return random.choice(['U', 'D', 'L', 'R'])
        else:
            # Exploit: choose the best action based on current Q-values
            q_values = [self.q_table.get((state, action), 0) for action in ['U', 'D', 'L', 'R']]
            max_q = max(q_values)
            # If multiple actions have the same Q-value, choose randomly among them
            actions = [action for action, q in zip(['U', 'D', 'L', 'R'], q_values) if q == max_q]
            return random.choice(actions)

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, a), 0) for a in ['U', 'D', 'L', 'R']])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def reset_game(self):
        super().reset_game()
        self.moves_left = self.MAX_MOVES_PER_GAME  # Reset moves left at the start of each game
        self.food_eaten = 0  # Reset the count of food eaten

    def train_step(self):
        state = self.get_state()
        action = self.get_action(state)
        score_before = self.score
        _, game_over = self.play_step(action)
        score_after = self.score

        # Update moves left and reward function
        if score_after > score_before:
            self.food_eaten += 1  # Increase food eaten count
            reward = 1000 * self.score  # Increase reward for each food eaten
            self.moves_left += 50  # Grant extra 50 moves
        elif game_over:
            if self.snake[0] in self.snake[1:]:
                reward = -10000  # Large negative reward for running into itself
            else:
                reward = -10000  # Negative reward for losing the game by other means
        else:
            reward = -25  # Small negative reward for each move

        self.moves_left -= 1  # Decrease moves left for each move

        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)

        # End the game if no moves are left
        if self.moves_left <= 0:
            return True
        return game_over


    def save_brain(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)


# Training loop with visualization, episode, and moves left display
ai_game = SnakeGameAI()
train_episodes = 10000
visualize_every = 1000

for episode in range(train_episodes):
    game_over = False
    ai_game.reset_game()  # Ensure game state is reset at the start of each episode
    while not game_over:
        game_over = ai_game.train_step()
        if episode % visualize_every == 0:
            ai_game.draw_board(episode)
    ai_game.epsilon *= ai_game.epsilon_decay
    ai_game.scores.append(ai_game.score)  # Store score for this episode
    if episode % visualize_every == 0:
        plt.pause(0.001)


# After the training loop
# Display top ten generations and their scores
top_scores = sorted([(score, ep + 1) for ep, score in enumerate(ai_game.scores)], reverse=True)[:10]
plt.figure(figsize=(10, 6))
plt.title("Top 10 Generations and Their Scores")
plt.bar(range(1, 11), [score for score, _ in top_scores], tick_label=[f"Gen {ep}" for _, ep in top_scores])
plt.xlabel("Generation")
plt.ylabel("Score")
plt.show(block=True)


# After training loop or at the point of saving
ai_game.save_brain("best_snake_brain.pkl")
