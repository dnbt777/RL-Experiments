
import pygame

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Game settings
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30

# Display settings
SCREEN_WIDTH = BOARD_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]

# RL settings
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# Training settings
EPISODES = 10000
MAX_STEPS = 1000

# Inference settings
NUM_GAMES = 5
RECORD_FILE = "tetris_gameplay.pkl"

# Pygame initialization
pygame.init()
FONT = pygame.font.Font(None, 36)
