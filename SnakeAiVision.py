import random
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

# Use the Tkinter backend for matplotlib, suitable for interactive plots
matplotlib.use('TkAgg')

# Basic SnakeGame class
class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset_game()

    def reset_game(self):
        self.snake = deque([(self.width//2, self.height//2)])
        self.score = 0
        self.food = None
        self.place_food()
        self.game_over = False

    def place_food(self):
        while self.food is None or self.food in self.snake:
            self.food = (random.randint(0, self.width-1), random.randint(0, self.height-1))

    def play_step(self, action):
    assert action in ['U', 'D', 'L', 'R'], "Invalid action."
    x, y = self.snake[0]
    if action == 'U': y -= 1
    if action == 'D': y += 1
    if action == 'L': x -= 1
    if action == 'R': x += 1
    self.snake.appendleft((x, y))

    # Corrected line here
    if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in set(list(self.snake)[1:]):
        self.game_over = True
        return self.score, self.game_over

    if (x, y) == self.food:
        self.score += 1
        self.place_food()
    else:
        self.snake.pop()

    return self.score, self.game_over


# SnakeGameVisual class for visual representation
class SnakeGameVisual(SnakeGame):
    def __init__(self, width=10, height=10):
        super().__init__(width, height)
        plt.ion()

    def draw_board(self, episode=None, move_counter=None):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.clear()
        self.ax.set_xlim(-1, self.width)
        self.ax.set_ylim(-1, self.height)
        self.ax.grid(False)
        self.ax.set_xticks(range(self.width))
        self.ax.set_yticks(range(self.height))

        title = f"Score: {self.score}"
        if episode is not None:
            title += f" | Episode: {episode}"
        self.ax.set_title(title)

        for x, y in self.snake:
            snake_rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, edgecolor='black', facecolor='green')
            self.ax.add_patch(snake_rect)

        fx, fy = self.food
        food_rect = patches.Rectangle((fx-0.5, fy-0.5), 1, 1, edgecolor='black', facecolor='red')
        self.ax.add_patch(food_rect)

        plt.draw()
        plt.pause(0.1)

# SnakeGameAI class with OpenCV for decision making
class SnakeGameAIWithVision(SnakeGameVisual):
    def __init__(self, width=10, height=10):
        super().__init__(width, height)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def make_decision_based_on_image(self, processed_data):
        decision = 'U' if np.mean(processed_data) > 128 else 'D'
        return decision

    def train_step_with_vision(self):
        image = self.capture_image()
        processed_data = self.process_image(image) if image is not None else None

        if processed_data is not None:
            action = self.make_decision_based_on_image(processed_data)
        else:
            action = random.choice(['U', 'D', 'L', 'R'])

        _, game_over = self.play_step(action)
        return game_over

# Main game loop
ai_game = SnakeGameAIWithVision(width=10, height=10)
train_episodes = 100  # Reduced for demonstration purposes

for episode in range(train_episodes):
    game_over = False
    ai_game.reset_game()
    while not game_over:
        game_over = ai_game.train_step_with_vision()
        ai_game.draw_board(episode)

# After the training loop
plt.show(block=True)
