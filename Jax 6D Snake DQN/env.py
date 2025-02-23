from typing import NamedTuple, List, Tuple
import jax.numpy as jnp
import jax.random as jrand
import jax
import time

## Improvements:
# Pytorch/pygame => Pure Jax (compiled)
# Single env => batched envs
# Readable code, namedtuples, mypy/typing
# fp32 => fp16 - more memory efficient

# Structure of Arrays
# batched env state => extremely parallelized env
class GameState(NamedTuple):
    apples : jax.Array
    apple_count : int
    snake_tail : jax.Array
    snake_tail_length : int
    snake_head : jax.Array

class GameStateBatch(NamedTuple):
    apples : jax.Array
    apple_count : List[int]
    snake_tail : jax.Array
    snake_tail_length : List[int]
    snake_head : jax.Array

## env.py
# 2D grid
# 0 is empty, 1 is snake head, 2 is snake body, 3 is apple
BATCH_SIZE = 128
HEIGHT = 10
WIDTH = 10
APPLE_COUNT = 5 # for now.. keep constant
DIRECTIONS = jnp.array([
  [-1, 0], # up
  [1, 0],  # down
  [0, -1], # left
  [0, 1]   # right
  ])


def init_game_state(
    key,
    height: int = HEIGHT,
    width: int = WIDTH,
    apple_count: int = APPLE_COUNT
    ) -> GameState:
  total_cells : int = height*width
  # randomly spawn apples and player
  spawn_cell_indices = jrand.choice(key, total_cells, (apple_count + 1,), replace=False)
  # apples
  apple_indices = spawn_cell_indices[1:]
  apple_rows = apple_indices // width
  apple_columns = apple_indices % width
  apples = jnp.stack([apple_rows, apple_columns], axis=-1, dtype=int)
  # player
  max_snake_len = total_cells
  snake_tail = -jnp.ones((max_snake_len - 1, 2), dtype=int)
  snake_head_index = spawn_cell_indices[0]
  snake_head_row = snake_head_index // width
  snake_head_column = snake_head_index % width
  snake_head = jnp.array([snake_head_row, snake_head_column], dtype=int)

  state = GameState(
    apples = apples,
    apple_count = apple_count,
    snake_tail = snake_tail,
    snake_tail_length = 0,
    snake_head = snake_head,
  )
  return state


# converts an array of structs into a struct of arrays
# useful for creating SoAs whose attributes jax can easily batch over
def aos_to_soa(aos: GameState) -> GameStateBatch:
  soa = jax.tree_util.tree_map(lambda *attr: jnp.stack(attr), aos)
  return soa

# converts soa to aos
# useful for replay buffer
def soa_to_aos(soa: GameStateBatch) -> GameState:
  return jax.vmap(GameState)(soa)


def init_game_state_batch(
    key,
    batchsize: int = BATCH_SIZE,
    height: int = HEIGHT,
    width: int = WIDTH,
    apple_count: int = APPLE_COUNT
    ) -> GameStateBatch:
  keys = jrand.split(key, batchsize)
  game_states_aos = jax.vmap(init_game_state, in_axes=(0, None, None, None))(keys, height, width, apple_count) # array of structures
  game_states_soa = aos_to_soa(game_states_aos)
  return game_states_soa


def cell_is_deadly(
    game_state: GameState,
    cell: jax.Array,
    height: int = HEIGHT,
    width: int = WIDTH
  ) -> bool:
  # cell is in tail -> death
  cell_is_in_tail = jnp.all(cell == game_state.snake_tail, axis=-1)
  # cell collides with wall -> death
  cell_is_in_wall = jnp.array([
    cell[0] < 0,
    cell[0] >= width,
    cell[1] < 0,
    cell[1] >= height
  ])
  is_deadly = jnp.any(jnp.concatenate([cell_is_in_tail, cell_is_in_wall]))
  return is_deadly


# update grid (state, move)
# returns: state, reward, finished
def update_game_state(
    game_state: GameState,
    action: int,
    height: int = HEIGHT,
    width: int = WIDTH
    ) -> Tuple[GameState, float, bool]:
  ## init action:
  # 0, 1, 2, 3 => up, down, left, right
  direction = DIRECTIONS[action]

  ### process current state
  ### move the snake, check for death, check for apples.

  ## init reward
  reward = 0.1 # for surviving
  finished = False # False by default

  ## update snake
  # update head
  new_snake_head = game_state.snake_head + direction
  # update snake body/tail - set last index to new
  new_snake_tail = game_state.snake_tail.at[game_state.snake_tail_length].set(game_state.snake_head)
  # new head touches apple -> :D
  eaten_apples = jnp.all(new_snake_head == game_state.apples, axis=1)
  if jnp.any(eaten_apples):
    reward += 1
    new_apples = game_state.apples.at[eaten_apples].set(jnp.array([0, 0])) # just reset it to 0, 0 for now idk
    new_snake_tail_length = game_state.snake_tail_length + 1
    # TODO add new apple randomly
  else:
    # get rid of tail if the snake didnt eat an apple
    # wait, since when do snakes eat apples?
    new_snake_tail = jnp.roll(new_snake_tail, -1, axis=0).at[-1].set(jnp.array([-1, -1]))
    new_snake_tail_length = game_state.snake_tail_length
    # apples are unchanged
    new_apples = game_state.apples

  ## rewards/punishments:
  # new head collides with tail -> death
  if cell_is_deadly(game_state, new_snake_head):
    finished = True
    reward -= 10

  next_state = GameState(
    apples = new_apples,
    apple_count = game_state.apple_count,
    snake_tail = new_snake_tail,
    snake_tail_length = new_snake_tail_length,
    snake_head = new_snake_head,
  )
  return next_state, reward, finished


def update_game_state_batch(
    game_state_batch: GameStateBatch,
    actions: List[int],
    height: int = HEIGHT,
    width: int = WIDTH
    ) -> Tuple[GameStateBatch, List[float], List[bool]]:
  return jax.vmap(update_game_state, in_axes=(0, 0, None, None))(game_state_batch, actions, height, width) # mypy doesnt understand vmap


def get_grid(
    game_state: GameState,
    height: int = HEIGHT,
    width: int = WIDTH
    ) -> jax.Array:
  grid = jnp.zeros((width, height))
  # set head cell to 1
  head_row, head_col = game_state.snake_head[0], game_state.snake_head[1]
  grid = grid.at[head_row, head_col].set(1)
  # set tail cells to 2
  tail_cell_mask = jnp.all(game_state.snake_tail != -1, axis=1)
  tail_cells = game_state.snake_tail[tail_cell_mask]
  tail_rows, tail_columns = tail_cells[:, 0], tail_cells[:, 1]
  grid = grid.at[tail_rows, tail_columns].set(2)
  # set apple cells to 3
  apple_rows, apple_columns = game_state.apples[:, 0], game_state.apples[:, 1]
  grid = grid.at[apple_rows, apple_columns].set(3)
  return grid










