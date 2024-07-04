
import numpy as np
import random
from config import *

class TetrisEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_piece = None
        self.current_pos = None
        self.score = 0
        self.game_over = False

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_piece = self._get_new_piece()
        self.current_pos = [0, BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.score = 0
        self.game_over = False
        return self._get_state()

    def step(self, action):
        if self.game_over:
            return self._get_state(), 0, True, {}

        reward = 0
        if action == 0:  # Move left
            self._move(-1)
        elif action == 1:  # Move right
            self._move(1)
        elif action == 2:  # Rotate
            self._rotate()
        elif action == 3:  # Drop
            reward = self._drop()

        if self._check_collision():
            self._place_piece()
            lines_cleared = self._clear_lines()
            reward += lines_cleared ** 2 * 10
            self.current_piece = self._get_new_piece()
            self.current_pos = [0, BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]

            if self._check_collision():
                self.game_over = True
                reward -= 50

        return self._get_state(), reward, self.game_over, {}

    def _get_state(self):
        state = self.board.copy()
        for y, row in enumerate(self.current_piece):
            for x, val in enumerate(row):
                if val:
                    state[self.current_pos[0] + y][self.current_pos[1] + x] = 2
        return state.flatten()

    def _get_new_piece(self):
        return random.choice(SHAPES)

    def _move(self, dx):
        self.current_pos[1] += dx
        if self._check_collision():
            self.current_pos[1] -= dx

    def _rotate(self):
        rotated = list(zip(*self.current_piece[::-1]))
        if not self._check_collision(rotated):
            self.current_piece = rotated

    def _drop(self):
        drop_distance = 0
        while not self._check_collision():
            self.current_pos[0] += 1
            drop_distance += 1
        self.current_pos[0] -= 1
        return drop_distance

    def _check_collision(self, piece=None):
        if piece is None:
            piece = self.current_piece
        for y, row in enumerate(piece):
            for x, val in enumerate(row):
                if val:
                    dy, dx = self.current_pos[0] + y, self.current_pos[1] + x
                    if dy >= BOARD_HEIGHT or dx < 0 or dx >= BOARD_WIDTH or (dy >= 0 and self.board[dy][dx]):
                        return True
        return False

    def _place_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, val in enumerate(row):
                if val:
                    self.board[self.current_pos[0] + y][self.current_pos[1] + x] = 1

    def _clear_lines(self):
        lines_cleared = 0
        new_board = []
        for row in self.board:
            if all(row):
                lines_cleared += 1
            else:
                new_board.append(row)
        self.board = np.array(([0] * BOARD_WIDTH for _ in range(lines_cleared)) + new_board)
        return lines_cleared
