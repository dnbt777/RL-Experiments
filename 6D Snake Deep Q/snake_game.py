
import random
from config import *
import numpy as np
from utils import *

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2, NUM_GRIDS_W // 2, NUM_GRIDS_V // 2, NUM_GRIDS_U // 2)]
        self.direction = random.choice(BASIS_DIRECTIONS)
        self.apples = self.place_food()
        self.done = False
        self.death_type = None
        self.net_reward = 0
        self.steps_without_apple = 0
        self.target_apple = self.get_random_apple()
        self.previous_distance = self.get_distance_to_target_apple()
        self.previous_coords = None
        self.average_position = np.array(self.snake[0], dtype=float)
        self.ewma_decay = 0.99
        self.consecutive_steps_away_from_apple = 0
        self.consecutive_steps_towards_apple = 0
        self.steps_towards_apple = 0
        self.steps_away_from_apple = 0
        self.target_apple_unit_direction = None
        self.apples_collected = 0
        return self.get_state()

    def place_food(self):
        apples = []
        while len(apples) < APPLE_COUNT:
            apple = (random.randint(0, GRID_SIZE - 1),
                     random.randint(0, GRID_SIZE - 1),
                     random.randint(0, GRID_SIZE - 1),
                     random.randint(0, NUM_GRIDS_W - 1),
                     random.randint(0, NUM_GRIDS_V - 1),
                     random.randint(0, NUM_GRIDS_U - 1))
            if apple not in self.snake and apple not in apples:
                apples.append(apple)
        return apples

    def get_random_apple(self):
        return random.choice(self.apples)

    def get_distance_to_target_apple(self):
        return sum(abs(h - a) for h, a in zip(self.snake[0], self.target_apple))

    def get_state(self):
        head = self.snake[0]
        
        target_apple_unit_direction = [unit(a - h) for a, h in zip(self.target_apple, head)]
        self.target_apple_unit_direction = target_apple_unit_direction
        
        surrounding_cells = [
            0 if not (any(
                    coord < 0 or coord >= GRID_SIZE 
                    for coord in cell[:3]) or
                      
                    cell[3] < 0 or cell[3] >= NUM_GRIDS_W or
                    cell[4] < 0 or cell[4] >= NUM_GRIDS_V or
                    cell[5] < 0 or cell[5] >= NUM_GRIDS_U or
                    cell in self.snake[1:]
            )
            else 1
            for cell in (tuple(h + d for h, d in zip(head, direction)) for direction in BASIS_DIRECTIONS)
        ]
        
        # return np.array(target_apple_unit_direction, dtype=np.float32) # debug
        return np.array(target_apple_unit_direction + surrounding_cells, dtype=np.int8)





    def step(self, action):
        head = self.snake[0]
        new_direction = BASIS_DIRECTIONS[action]
        new_head = tuple(h + d for h, d in zip(head, new_direction))
        

        reward = 0.0 # give it a bit for surviving

        if (any(coord < 0 or coord >= GRID_SIZE for coord in new_head[:3]) or
            new_head[3] < 0 or new_head[3] >= NUM_GRIDS_W or
            new_head[4] < 0 or new_head[4] >= NUM_GRIDS_V or
            new_head[5] < 0 or new_head[5] >= NUM_GRIDS_U or
            new_head in self.snake):
            self.done = True
            reward = -2
            self.death_type = "wall collision" if new_head not in self.snake else "self collision"
        else:
            self.snake.insert(0, new_head)
            self.direction = new_direction

            self.average_position = self.ewma_decay * self.average_position + (1 - self.ewma_decay) * np.array(new_head)

            if new_head == self.target_apple:
                reward += 0
                self.apples_collected += 1
                self.apples.remove(self.target_apple)
                self.apples.append(self.place_food()[0])
                self.target_apple = self.get_random_apple()
            else:
                self.snake.pop()

            current_distance = self.get_distance_to_target_apple()
            if current_distance < self.previous_distance:
                reward += 0.2
                self.steps_towards_apple += 1
            else:
                reward -= 0.2
                self.steps_away_from_apple += 1
                #reward -= self.steps_away_from_apple/100
            self.previous_distance = current_distance

            if self.steps_towards_apple + self.steps_away_from_apple > 5000:
                self.death_type = "stopped"
                self.done=True

        self.net_reward += reward

        return self.get_state(), reward, self.done

    def get_render_data(self):
        return self.snake, self.apples
