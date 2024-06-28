
import pygame
import random
from config import *
from math import *

def abslogint(x):
    if x == 0:
        y = 0
    else:
        y = copysign(1, x)*abs(int(1+2*log(abs(x))))
    return y

class SnakeGame:
    def __init__(self, screen):
        self.screen = screen
        self.reset()

    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.food = self.place_food()
        self.done = False
        self.previous_distance = self.get_distance_to_food()
        return self.get_state()

    def place_food(self):
        while True:
            x = random.randint(0, (WIDTH // CELL_SIZE) - 1) * CELL_SIZE
            y = random.randint(0, (HEIGHT // CELL_SIZE) - 1) * CELL_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def get_distance_to_food(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def get_state(self):
        head = self.snake[0]
        food = self.food
        
        # Distance to walls
        distance_to_left = head[0] // CELL_SIZE
        distance_to_right = (WIDTH - head[0]) // CELL_SIZE
        distance_to_top = head[1] // CELL_SIZE
        distance_to_bottom = (HEIGHT - head[1]) // CELL_SIZE
        
        # Relative position of the food to the head
        food_rel_x = (food[0] - head[0]) // CELL_SIZE
        food_rel_y = (food[1] - head[1]) // CELL_SIZE
        
        # Check for tail in adjacent cells
        adjacent_cells = [
            (head[0], head[1] - CELL_SIZE),  # Up
            (head[0], head[1] + CELL_SIZE),  # Down
            (head[0] - CELL_SIZE, head[1]),  # Left
            (head[0] + CELL_SIZE, head[1])   # Right
        ]
        
        tail_state = 0
        for i, cell in enumerate(adjacent_cells):
            if cell in self.snake[1:]:
                tail_state |= (1 << (3 - i))
        
        return (
            abslogint(distance_to_left),
            abslogint(distance_to_right),
            abslogint(distance_to_top),
            abslogint(distance_to_bottom),
            abslogint(food_rel_x),
            abslogint(food_rel_y),
            tail_state
        )

    def step(self, action):
        new_direction = self.direction
        if action == 0:  # up
            new_direction = (0, -1)
        elif action == 1:  # down
            new_direction = (0, 1)
        elif action == 2:  # left
            new_direction = (-1, 0)
        elif action == 3:  # right
            new_direction = (1, 0)

        # Check if the new direction would cause the snake to move backwards
        if len(self.snake) > 1:
            head = self.snake[0]
            neck = self.snake[1]
            if (head[0] + new_direction[0] * CELL_SIZE, head[1] + new_direction[1] * CELL_SIZE) == neck:
                # If moving backwards, keep the current direction
                new_direction = self.direction

        self.direction = new_direction
        head = self.snake[0]
        new_head = (head[0] + self.direction[0] * CELL_SIZE, head[1] + self.direction[1] * CELL_SIZE)

        # Check for collision with walls or self
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= WIDTH or 
            new_head[1] < 0 or new_head[1] >= HEIGHT):
            self.done = True
            return self.get_state(), -100, self.done  # Massive negative reward for death

        self.snake.insert(0, new_head)

        if new_head == self.food:
            reward = 10  # Large positive reward for eating food
            self.food = self.place_food()
            self.previous_distance = self.get_distance_to_food()
        else:
            self.snake.pop()
            new_distance = self.get_distance_to_food()
            if new_distance < self.previous_distance:
                reward = 1  # Small positive reward for moving towards food
            else:
                reward = -2  # Small negative reward for moving away from food
            self.previous_distance = new_distance

        return self.get_state(), reward, self.done

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0], self.food[1], CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
