from env import *
import jax.random as jrand


# render.py

# init game state
key = jrand.PRNGKey(0)
batchsize = 2
game_state_batch = init_game_state_batch(key, batchsize=batchsize)

for i in range(100):
  # print grid
  grid = get_grid_batched(game_state_batch)
  print(grid)

  # get action from user/model
  action_batch = int(input("Enter action: (u/d/l/r 0,1,2,3)")) * jnp.ones((batchsize,), dtype=int)

  # get next game state
  transition_step_batch = update_game_state_batched(game_state_batch, action_batch)
  game_state_batch = transition_step_batch.next_state
  finished = transition_step_batch.finished
  if jnp.any(finished):
    print("GAME OVER")
    break