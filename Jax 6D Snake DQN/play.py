from env import *
from dqn import *
import jax.random as jrand

# init game state
rolling_key: jax.Array = jrand.PRNGKey(0)
batchsize: int = 1
game_state_batch: GameStateBatch = init_game_state_batch(rolling_key, batchsize=batchsize)
rolling_key, _ = jrand.split(rolling_key, 2)

#jax.config.update("jax_disable_jit", True)

for i in range(100):
  # print grid
  grid: jax.Array = get_grid_batched(game_state_batch)
  print(grid)
  print("[[Y X U D L R]]")
  model_vision: jax.Array = get_model_vision_batched(game_state_batch)
  print(model_vision)

  # get chosen action from user/model
  action: int = dict(zip("wsad", range(4)))[input("WASD:")]
  action_batch: jax.Array = int(action)*jnp.ones((batchsize,), dtype=int)

  # get next game state
  rolling_key, _ = jrand.split(rolling_key, 2)
  transition_step_batch: TransitionStepBatch = update_game_state_batched(rolling_key, game_state_batch, action_batch)
  game_state_batch = transition_step_batch.next_state_batch
  finished_batch: jax.Array = transition_step_batch.finished_batch
  print("reward: ", transition_step_batch.reward_batch)

  if jnp.any(finished_batch):
    print("GAME OVER")
    break