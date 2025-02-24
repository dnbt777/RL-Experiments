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

MODEL_DTYPE=jnp.float32 # float16 would be better. or bf16


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

@jax.jit
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


@jax.jit
def get_model_vision_batched(game_state_batch : GameStateBatch) -> jax.Array:
  # get direction towards first apple in the list
  # ### TODO INVESTIGATE IF THIS REALLY WORKS (test it?)
  relative_apple_direction = jnp.sign(game_state_batch.apples[:, 0] - game_state_batch.snake_head)
  # which directions cause death?
  batch_size = game_state_batch.snake_head.shape[0]
  cells_around_snake_head_batch = jnp.tile(game_state_batch.snake_head, 4).reshape(batch_size, 4, 2) + DIRECTIONS
  deadly_directions_batch = jax.vmap(cell_is_deadly_batched, in_axes=(None, 1))(game_state_batch, cells_around_snake_head_batch)
  deadly_directions_batch = jnp.transpose(deadly_directions_batch)
  
  # concat and return what the model can see
  model_vision = jnp.concat([relative_apple_direction, deadly_directions_batch], axis=-1)
  return model_vision


@functools.partial(jax.jit, static_argnames=["dtype"])
def get_action_qualities_batched(
    model_params: MLPParams,
    game_state_batched: GameStateBatch,
    dtype=MODEL_DTYPE
    ) -> jax.Array:
  model_vision = get_model_vision_batched(game_state_batched)
  model_input = jnp.array(model_vision, dtype=dtype)
  qualities = mlp_forward(model_params, model_input)
  return qualities


# take_action(model_params, state)
# returns: action
@functools.partial(jax.jit, static_argnames=["dtype", "epsilon"])
def take_action_batched(
    model_params: MLPParams,
    game_state_batch: GameStateBatch,
    key: jax.Array,
    epsilon: float = 0,
    dtype=MODEL_DTYPE
    ) -> jax.Array:
  qualities_batch = get_action_qualities_batched(model_params, game_state_batch, dtype=dtype)

  batch_size = game_state_batch.snake_head.shape[0]
  action_count = qualities_batch.shape[-1]
  rand_chance_batch = jrand.uniform(key, shape=(batch_size,))
  key, _ = jrand.split(key, 2)
  action_batch = jnp.where(
    rand_chance_batch < epsilon,
    jrand.randint(key, (batch_size,), minval=0, maxval=action_count), # maxval is exclusive here
    jnp.argmax(qualities_batch, axis=-1)
  )
  return action_batch


@jax.jit
def mse(yhat, y):
  return jnp.mean((jnp.array(y) - jnp.array(yhat))**2)