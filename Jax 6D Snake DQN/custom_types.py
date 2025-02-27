from typing import NamedTuple, List, Tuple
import jax

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

# stores transition steps. Same structure as TransitionStepBatch.
# In the future, maybe these types can be merged, but for now this makes the code easier to reason about.
class ReplayBuffer(NamedTuple):
    state_batch: GameStateBatch     # (batch_size, *)
    action_batch: jax.Array      # (batch_size,)
    reward_batch: jax.Array      # (batch_size,)
    next_state_batch: GameStateBatch  # (batch_size, *)
    finished_batch: jax.Array          # (batch_size,)


# DQN

class HiddenLayer(NamedTuple):
  w: jax.Array
  b: jax.Array

class MLPParams(NamedTuple):
  wi: jax.Array
  bi: jax.Array
  wo: jax.Array
  bo: jax.Array
  hidden_layers: List[HiddenLayer]
