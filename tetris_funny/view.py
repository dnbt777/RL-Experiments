
import pygame
import pickle
import time
from config import *

def draw_board(screen, board):
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell == 1:
                pygame.draw.rect(screen, WHITE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            elif cell == 2:
                pygame.draw.rect(screen, RED, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

def view_gameplay():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris Replay")

    with open(RECORD_FILE, 'rb') as f:
        gameplay_data = pickle.load(f)

    for game in gameplay_data:
        print(f"Viewing Game {game['game_number']}: Score = {game['score']}")
        for state in game['states']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill(BLACK)
            draw_board(screen, state)
            score_text = FONT.render(f"Score: {game['score']}", True, WHITE)
            screen.blit(score_text, (10, 10))
            pygame.display.flip()
            time.sleep(0.1)

        time.sleep(1)  # Pause between games

    pygame.quit()

if __name__ == "__main__":
    view_gameplay()
