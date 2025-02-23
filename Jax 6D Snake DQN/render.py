from env import *
import jax.random as jrand


# render.py

# init game state
key = jrand.PRNGKey(0)
game_state = init_game_state(key)

for i in range(100):
  # print grid
  grid = get_grid(game_state)
  print(grid)

  # get action from user/model
  action = int(input("Enter action: (u/d/l/r 0,1,2,3)"))

  # get next game state
  game_state, reward, finished = update_game_state(game_state, action)
  if finished:
    print("GAME OVER")
    break