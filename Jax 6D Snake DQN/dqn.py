## dqn.py
from env import *

from typing import NamedTuple, List
from functools import *
import jax
import jax.numpy as jnp

class HiddenLayer(NamedTuple):
  w: jax.Array
  b: jax.Array

class MLPParams(NamedTuple):
  wi: jax.Array
  bi: jax.Array
  wo: jax.Array
  bo: jax.Array
  hidden_layers: List[HiddenLayer]

MODEL_DTYPE=jnp.float16


def init_mlp_dqn(
    key: jax.Array,
    input_size: int,
    output_size: int,
    hidden_layers: int,
    hidden_layer_size: int,
    dtype=MODEL_DTYPE
    ) -> MLPParams:
  initializer = jax.nn.initializers.glorot_uniform()
  return MLPParams(
    wi=initializer(key, (input_size, hidden_layer_size), dtype=dtype),
    bi=jnp.zeros((hidden_layer_size,), dtype=dtype),
    wo=initializer(key, (hidden_layer_size, output_size), dtype=dtype),
    bo=jnp.zeros((output_size), dtype=dtype),
    hidden_layers=HiddenLayer(
      w=hidden_layers * initializer(key, (hidden_layers, hidden_layer_size, hidden_layer_size), dtype=dtype), # batched SoA, will be scanned across
      b=jnp.zeros((hidden_layers, hidden_layer_size), dtype=dtype)
    ) # mypy doesn't understand SoA with List type D:
  )


def mlp_forward(mlp_params: MLPParams, x: jax.Array) -> jax.Array:
  # project to hidden size
  x = jax.nn.relu(x @ mlp_params.wi + mlp_params.bi)
  # scan through hidden layers
  # scanf :: carry -> input_i -> carry -> output_i
  scanf = lambda _x, hidden_layer: (jax.nn.relu(_x @ hidden_layer.w + hidden_layer.b), None)
  x, _ = jax.lax.scan(scanf, x, mlp_params.hidden_layers)
  # project to output size
  x = jax.nn.relu(x @ mlp_params.wo + mlp_params.bo)
  return x


def get_model_vision(game_state : GameState) -> jax.Array:
  # get direction towards first apple in the list
  relative_apple_direction = jnp.sign(game_state.apples[0] - game_state.snake_head)
  # which directions cause death?
  cells_around_snake_head = jnp.array([game_state.snake_head + direction for direction in DIRECTIONS]) # probably a more elegant, jax-esque way to do this
  deadly_direction_mask = jax.vmap(lambda cell: cell_is_deadly(game_state, cell), in_axes=0)(cells_around_snake_head)
  # concat and return what the model can see
  model_vision = jnp.concat([relative_apple_direction, deadly_direction_mask])
  return model_vision


def get_action_qualities(
    model_params: MLPParams,
    game_state: GameState,
    dtype=MODEL_DTYPE
    ) -> jax.Array:
  model_vision = get_model_vision(game_state)
  model_input = jnp.array(model_vision, dtype=dtype)
  qualities = mlp_forward(model_params, model_input)
  return qualities


# take_action(model_params, state)
# returns: action
def take_action(
    model_params: MLPParams,
    game_state: GameState,
    key: jax.Array,
    epsilon: float = 0,
    dtype=MODEL_DTYPE
    ) -> int:
  qualities = get_action_qualities(model_params, game_state, dtype=dtype)
  # ITS NOT LEARNING BC MY KEY ISNT ROLLING
  rand_chance = jrand.uniform(key, shape=(1,))[0]
  if rand_chance < epsilon:
    key, _ = jrand.split(key, 2)
    action = jrand.randint(key, (1,), 0, qualities.shape[0])[0]
  else:
    action = jnp.argmax(qualities) # ignore key for now
  return action



def mse(yhat, y):
  return jnp.mean(jnp.array(y)**2 - jnp.array(yhat)**2)