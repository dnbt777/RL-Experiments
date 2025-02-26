from typing import NamedTuple, List, Tuple
import functools
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
class GameStateBatch(NamedTuple):
    apples : jax.Array
    score : jax.Array
    snake_tail : jax.Array
    snake_tail_length : jax.Array
    snake_head : jax.Array

# TransitionStep = Tuple(game_state, action, reward, next_state, finished)
# TransitionStepBatch = Tuple(game_state_batch, action_batch, reward_batch, next_state_batch, finished_batch)
class TransitionStepBatch(NamedTuple):
    state_batch: GameStateBatch
    action_batch: jax.Array
    reward_batch: jax.Array
    next_state_batch: GameStateBatch
    finished_batch: jax.Array

## env.py
# 2D grid
# 0 is empty, 1 is snake head, 2 is snake body, 3 is apple
BATCH_SIZE = 128
HEIGHT = 10
WIDTH = 10
APPLE_COUNT = 1 # for now.. keep constant
DIRECTIONS = jnp.array([
  # y, x
  [1, 0], # up
  [-1, 0],  # down
  [0, -1], # left
  [0, 1]   # right
  ])



@functools.partial(jax.jit, static_argnames=["height", "width", "apple_count", "batchsize"])
def init_game_state_batch(
    key,
    batchsize: int = BATCH_SIZE,
    height: int = HEIGHT,
    width: int = WIDTH,
    apple_count: int = APPLE_COUNT
    ) -> GameStateBatch:
  total_cells: int = height*width
  # randomly spawn apples and player
  batch_keys = jrand.split(key, batchsize)
  spawn_cell_indices = jax.vmap(jrand.choice, in_axes=(0, None, None))(batch_keys, total_cells, (apple_count + 1,))
  # apples
  apple_indices = spawn_cell_indices[:, 1:]
  apple_rows = apple_indices // width
  apple_columns = apple_indices % width
  apples = jnp.stack([apple_rows, apple_columns], axis=-1, dtype=int)
  # player
  max_snake_len = total_cells
  snake_tail = -jnp.ones((batchsize, max_snake_len - 1, 2), dtype=int)
  snake_head_index = spawn_cell_indices[:, 0]
  snake_head_rows = snake_head_index // width
  snake_head_columns = snake_head_index % width
  snake_head = jnp.stack([snake_head_rows, snake_head_columns], axis=-1, dtype=int)

  state_batch = GameStateBatch(
    apples = apples,
    score = jnp.zeros((batchsize,), dtype=int),
    snake_tail = snake_tail,
    snake_tail_length = jnp.zeros((batchsize,), dtype=int),
    snake_head = snake_head,
  )
  return state_batch


@functools.partial(jax.jit, static_argnames=["height", "width"])
def cell_is_deadly_batched(
    game_state_batch: GameStateBatch, # (batch, *) SoA
    cell_batch: jax.Array, # (batch, 2)
    height: int = HEIGHT,
    width: int = WIDTH
  ) -> jax.Array:
  # cell is in tail -> death
  cell_is_in_tail = jnp.all(cell_batch[:, jnp.newaxis, :] == game_state_batch.snake_tail, axis=-1)
  # cell collides with wall -> death
  cell_is_in_wall = jnp.swapaxes(jnp.array([
    # cell = y, x
    cell_batch[:, 0] < 0, 
    cell_batch[:, 0] >= height,
    cell_batch[:, 1] < 0,
    cell_batch[:, 1] >= width
  ]), 0, 1)
  is_deadly = jnp.any(jnp.concatenate([cell_is_in_tail, cell_is_in_wall], axis=-1), axis=-1)
  return is_deadly


# update grid (state, move)
# returns: state, reward, finished
@functools.partial(jax.jit, static_argnames=["height", "width"])
def update_game_state_batched(
    key: jax.Array,
    game_state_batch: GameStateBatch,
    action_batch: jax.Array,
    height: int = HEIGHT,
    width: int = WIDTH,
    ) -> TransitionStepBatch:
  batch_size = game_state_batch.apples.shape[0]
  ## init action:
  # 0, 1, 2, 3 => up, down, left, right
  direction_batch = DIRECTIONS[action_batch]

  ## init reward
  reward_batch = jnp.ones_like(action_batch, dtype=jnp.float16) * 0.1 # for surviving
  finished_batch = jnp.zeros_like(action_batch, dtype=bool) # False by default

  ## update snake
  # update head
  new_snake_head = game_state_batch.snake_head + direction_batch
  # update snake body/tail - set last index to new
  new_snake_tail = game_state_batch.snake_tail.at[:, game_state_batch.snake_tail_length].set(game_state_batch.snake_head)
  # new head touches apple -> :D
  eaten_apples = jnp.all(new_snake_head[:, jnp.newaxis, :] == game_state_batch.apples, axis=-1)

  ## rewards/punishments:
  # snake ate an apple
  snake_ate_apple = jnp.any(eaten_apples, axis=-1) # (batch,)
  # update reward
  reward_batch = reward_batch + 3*snake_ate_apple
  # replace eaten apples with new apples, in each batch
  grid_batch = jnp.ones((batch_size, height, width), dtype=bool)
  grid_batch_indices = jnp.arange(batch_size)[:, jnp.newaxis, jnp.newaxis]
  grid_batch = grid_batch.at[grid_batch_indices, game_state_batch.apples[:, :, 0], game_state_batch.apples[:, :, 1]].set(False)
  grid_batch = grid_batch.at[grid_batch_indices, game_state_batch.snake_tail[:, :, 0], game_state_batch.snake_tail[:, :, 1]].set(False)
  batch_idx, free_y, free_x = jnp.where(grid_batch, size=grid_batch.size, fill_value=0) # fill with 0, 0. TODO fix. replace rand choice w index maxval excluding -1s (maxval = sum of mask x < 0)
  free_positions = jnp.stack([free_x, free_y], axis=-1).reshape(batch_size, -1, 2)
  apple_count = game_state_batch.apples.shape[1]
  updated_apples_batch = jrand.choice(key, free_positions, shape=(apple_count,), replace=False, axis=1)
  new_apples = jnp.where(
    eaten_apples[:, :, jnp.newaxis],
    updated_apples_batch,
    game_state_batch.apples,
  )
  # increase snake tail size
  new_snake_tail_length = game_state_batch.snake_tail_length + 1*snake_ate_apple

  # get rid of tail's tip if the snake didnt eat an apple
  # use rolling and jnp.where for this
  # wait, since when do snakes eat apples?
  rolled_snake_tail = jnp.roll(new_snake_tail, -1, axis=1).at[:, -1].set(jnp.array([-1, -1])) # batch, tail_length, coords. roll along tail len dim
  new_snake_tail = jnp.where(
     snake_ate_apple[:, jnp.newaxis, jnp.newaxis],
     new_snake_tail,
     rolled_snake_tail
  )

  # new head collides with tail -> death
  snake_died = cell_is_deadly_batched(game_state_batch, new_snake_head)
  finished_batch = jnp.logical_or(finished_batch, snake_died)
  reward_batch = reward_batch - 10*snake_died

  # update score if not dead
  new_score_batch = game_state_batch.score + 1*snake_ate_apple*(~finished_batch)
  
  next_state_batch = GameStateBatch(
    apples = new_apples,
    score = new_score_batch,
    snake_tail = new_snake_tail,
    snake_tail_length = new_snake_tail_length,
    snake_head = new_snake_head,
  )

  transition_step_batch = TransitionStepBatch(
    state_batch=game_state_batch,
    action_batch=action_batch,
    reward_batch=reward_batch,
    next_state_batch=next_state_batch,
    finished_batch=finished_batch

  )
  return transition_step_batch


# gets the grid of a batch of gamestates
def get_grid_batched(
    game_state_batch: GameStateBatch,
    height: int = HEIGHT,
    width: int = WIDTH
    ) -> jax.Array:
  batchsize = game_state_batch.apples.shape[0]
  grid_batch = jnp.zeros((batchsize, width, height))
  # set head cell to 1
  head_rows, head_cols = game_state_batch.snake_head[:, 0], game_state_batch.snake_head[:, 1] # batch,
  head_batch_index = jnp.arange(head_rows.shape[0])
  grid_batch = grid_batch.at[head_batch_index, head_rows, head_cols].set(1)
  # set tail cells to 2
  tail_cell_mask = jnp.all(game_state_batch.snake_tail != -1, axis=-1)
  tail_rows, tail_columns = game_state_batch.snake_tail[:, :, 0], game_state_batch.snake_tail[:, :, 1] # batch, tail, coords
  tail_rows, tail_columns = tail_rows[tail_cell_mask], tail_columns[tail_cell_mask]
  tail_cell_batch_index = jnp.arange(tail_cell_mask.shape[0])
  if tail_rows.shape[0] > 0:
    grid_batch = grid_batch.at[tail_cell_batch_index, tail_rows, tail_columns].set(2)
  # set apple cells to 3
  apple_rows, apple_columns = game_state_batch.apples[:, :, 0], game_state_batch.apples[:, :, 1] # batch, apple, coords
  apple_batch_index = jnp.arange(apple_rows.shape[0]) # redundant. same batchsize for all, so batch_index can just be used.
  grid_batch = grid_batch.at[apple_batch_index[:, jnp.newaxis], apple_rows, apple_columns].set(3)
  return grid_batch[:, ::-1,] # flip grid vertically for human player


# for batched sims
# write _batched functions
# use a SoA
# do not use vmap except where necessary
# writing unbatched functions and vmapping them feels like a poor design choice


# no jax.jit:     steps/s:  952.99
# after jax.jit:  steps/s:  46303.45