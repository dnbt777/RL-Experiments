from env import *
from dqn import *

from typing import NamedTuple, List, Optional
import time

debug = False
if debug:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)


# stores transition steps. Same structure as TransitionStepBatch.
# In the future, maybe these types can be merged, but for now this makes the code easier to reason about.
class ReplayBuffer(NamedTuple):
    state_batch: GameStateBatch     # (batch_size, *)
    action_batch: jax.Array      # (batch_size,)
    reward_batch: jax.Array      # (batch_size,)
    next_state_batch: GameStateBatch  # (batch_size, *)
    finished_batch: jax.Array          # (batch_size,)


def init_replay_buffer(
        buffer_size: int,
        init_transition_step_batch: TransitionStepBatch
        ) -> ReplayBuffer:
    # initializes a replay buffer of size buffer_size
    # initializer with garbage data, which will be overridden
    # it is promised that the replay buffer will not be used until it is filled
    batch_size = init_transition_step_batch.action_batch.shape[0]
    apple_count = init_transition_step_batch.state_batch.apples.shape[1]
    snake_tail_max_len = init_transition_step_batch.state_batch.snake_tail.shape[1]

    init_game_state = GameStateBatch(
        apples=jnp.zeros((buffer_size*batch_size, apple_count, 2)), #(buffer_size*batch_size, apple_count, 2)
        score=jnp.zeros((buffer_size*batch_size,)),
        snake_tail=jnp.zeros((buffer_size*batch_size, snake_tail_max_len, 2)),
        snake_tail_length=jnp.zeros((buffer_size*batch_size,)),
        snake_head=jnp.zeros((buffer_size*batch_size, 2))
    )
    return ReplayBuffer(
        state_batch=init_game_state,
        action_batch=jnp.zeros((buffer_size*batch_size,)),
        reward_batch=jnp.zeros((buffer_size*batch_size,)),
        next_state_batch=init_game_state,
        finished_batch=jnp.zeros((buffer_size*batch_size,)),
    )

def tsb_to_replay_buffer(transition_step_batch: TransitionStepBatch) -> ReplayBuffer:
    return ReplayBuffer(
        state_batch=transition_step_batch.state_batch,
        action_batch=transition_step_batch.action_batch,
        reward_batch=transition_step_batch.reward_batch,
        next_state_batch=transition_step_batch.next_state_batch,
        finished_batch=transition_step_batch.finished_batch,
    )

def replay_buffer_to_tsb(replay_buffer: ReplayBuffer) -> TransitionStepBatch:
    return TransitionStepBatch(
        state_batch=replay_buffer.state_batch,
        action_batch=replay_buffer.action_batch,
        reward_batch=replay_buffer.reward_batch,
        next_state_batch=replay_buffer.next_state_batch,
        finished_batch=replay_buffer.finished_batch,
    )


# Append a TransitionStepBatch to the end of the replay buffer and override the first val with rolling
# batch_size could be calculated from inputs, but, passing it as an arg allows it to be a static arg when jitted
def append_replay_buffer(
        replay_buffer: ReplayBuffer,
        transition_step_batch: TransitionStepBatch,
        batch_size: int = BATCH_SIZE
        ) -> ReplayBuffer:
    # roll replay buffer batch_size to the left
    replay_buffer = jax.tree_util.tree_map(lambda buffer: jnp.roll(buffer, -batch_size, axis=0), replay_buffer) 
    # update the end of the buffer to the latest batch of values
    buffer_update_fn = lambda buffer_arr, tsb_arr: buffer_arr.at[-batch_size:].set(tsb_arr)
    transition_step_batch_as_buffer = tsb_to_replay_buffer(transition_step_batch)
    replay_buffer = jax.tree_util.tree_map(buffer_update_fn, replay_buffer, transition_step_batch_as_buffer)
    return replay_buffer


@jax.jit
def train_transition_minibatch(
        model_params: MLPParams,
        target_model_params: MLPParams,
        minibatch: ReplayBuffer,
        ) -> Tuple[MLPParams, float, MLPParams]:
    target_quality = jnp.max(get_action_qualities_batched(target_model_params, minibatch.next_state_batch), axis=-1)
    target_Qsa = minibatch.reward_batch + gamma * target_quality*(1 - minibatch.finished_batch) # bellman equation term
    # get estimated quality of current state
    # backprop on target_Qsa == predicted_Qsa (MSE)
    loss_fn = lambda _model_params, _target_Qsa, _game_state_batch: mse(_target_Qsa, jnp.max(get_action_qualities_batched(_model_params, _game_state_batch)))
    loss, grads = jax.value_and_grad(loss_fn)(model_params, target_Qsa, minibatch.state_batch)
    # accumulate grads
    # update model
    model_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate*g, model_params, grads)
    return model_params, loss, grads


@functools.partial(jax.jit, static_argnames=["epsilon", "steps"])
def generate_replays(
        replay_buffer,
        model_params,
        game_state_batch,
        rolling_key,
        steps,
        epsilon=0.9
        ) -> Tuple[ReplayBuffer, GameStateBatch, jax.Array, jax.Array, jax.Array]:
    mean_rewards = jnp.zeros((steps,))
    numbers_dead = jnp.zeros((steps,))
    mean_scores = jnp.zeros((steps,))
    # jitting this for-loop will make compilation time slower and increase memory
    # but this is better than not jitting this code, which is bottlenecking training
    # TODO: turn the for-loop into a scan
    for step in range(steps):
        # Take an action step
        rolling_key, _ = jrand.split(rolling_key, 2)
        action_batch = take_action_batched(model_params, game_state_batch, rolling_key, epsilon=epsilon)
        # get next state
        # get action, reward(q)
        rolling_key, _ = jrand.split(rolling_key, 2)
        transition_step_batch = update_game_state_batched(rolling_key, game_state_batch, action_batch)
        # store in replay buffer
        replay_buffer = append_replay_buffer(replay_buffer, transition_step_batch) # roll and append to end
        # update game state
        game_state_batch = transition_step_batch.next_state_batch
        # replace finished game states with new ones
        rolling_key, _ = jrand.split(rolling_key, 2)
        fresh_game_state_batch = init_game_state_batch(rolling_key)
        game_state_batch = GameStateBatch(
            apples = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.apples,
                game_state_batch.apples
            ),
            score = jnp.where(
                transition_step_batch.finished_batch[:],
                fresh_game_state_batch.score,
                game_state_batch.score
            ),
            snake_head = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis],
                fresh_game_state_batch.snake_head,
                game_state_batch.snake_head
            ),
            snake_tail = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.snake_tail,
                game_state_batch.snake_tail
            ),
            snake_tail_length = jnp.where(
                transition_step_batch.finished_batch[:],
                fresh_game_state_batch.snake_tail_length,
                game_state_batch.snake_tail_length
            )
        ) # clunky. optimize manually, or jit so the compiler can attempt to optimize this (update: jitted)
        mean_reward = jnp.mean(transition_step_batch.reward_batch)
        mean_rewards = mean_rewards.at[step].set(mean_reward)
        mean_score = jnp.mean(transition_step_batch.state_batch.score)
        mean_scores = mean_scores.at[step].set(mean_score)
        number_dead = jnp.sum(transition_step_batch.finished_batch)
        numbers_dead = numbers_dead.at[step].set(number_dead)
    return replay_buffer, game_state_batch, mean_rewards, numbers_dead, mean_scores



## TRAIN LOOP SETUP
rolling_key = jrand.PRNGKey(0)
# MODEL PARAMS
sample_game_states: GameStateBatch = init_game_state_batch(rolling_key)
input_size: int = get_model_vision_batched(sample_game_states).shape[-1]
output_size: int = 4 # up/dn/L/R
hidden_layers: int = 5
hidden_layer_size: int = 16
rolling_key, _ = jrand.split(rolling_key, 2)
model_params: MLPParams = init_mlp_dqn(rolling_key, input_size, output_size, hidden_layers, hidden_layer_size)
# RL PARAMS
epsilon: float = 1.0
epsilon_decay_linear: float = 1.0/1000
epsilon_decay_exponential: float = 1
gamma: float = 0.9
learning_rate: float = 1e-4
learning_rate_decay_exponential: float = 0.98
# REPLAY BUFFER SETUP
replay_buffer_size: int = 256 # number of batches (steps), not number of samples
replay_buffer_index: int = 0
# TRAIN LOOP PARAMS
iterations_per_model_update: int = 1
play_steps_per_iteration: int = 128 # autoregressive - how many batches to add to the replay buffer
grad_update_iterations: int = 128 # autoregressive - how many times to repeat minibatch sampling and training
minibatch_batches: int = 128 # batches
minibatch_size: int = minibatch_batches * BATCH_SIZE # samples
train_loop_iterations: int = 1000000

## TRAIN LOOP EXECUTION
# init loop
loss: float = 0
# run loop
for train_loop_iteration in range(train_loop_iterations):
    # every 100 iters (including at the start) update the loop
    if train_loop_iteration % iterations_per_model_update == 0:
        rolling_key, _ = jrand.split(rolling_key, 2) # reroll random key -> new key
        game_state_batch = init_game_state_batch(rolling_key)
        target_model_params = model_params
        # decay epsilon
        epsilon = epsilon - epsilon_decay_linear
        epsilon = epsilon * epsilon_decay_exponential
        epsilon = max(0, epsilon)
        learning_rate = learning_rate * learning_rate_decay_exponential
    if train_loop_iteration == 0:
        # initialize replay buffer
        action_batch = take_action_batched(model_params, game_state_batch, rolling_key, epsilon=epsilon)
        rolling_key, _ = jrand.split(rolling_key, 2)
        transition_step_batch = update_game_state_batched(rolling_key, game_state_batch, action_batch)
        replay_buffer: ReplayBuffer = init_replay_buffer(replay_buffer_size, transition_step_batch)
    # start the timer
    start_time = time.time()
    if train_loop_iteration == 0:
        steps = replay_buffer_size # fill replay buffer first
    else:
        steps = play_steps_per_iteration
    # generate transition steps and add to the replay buffer
    rolling_key, _ = jrand.split(rolling_key, 2)
    replay_buffer, game_state_batch, mean_rewards, numbers_dead, mean_scores = generate_replays(
        replay_buffer,
        model_params,
        game_state_batch,
        rolling_key,
        steps,
        epsilon=epsilon
        )
    print(f"generated_batches={steps}/mean_reward={jnp.mean(mean_rewards):0.2f}/deaths={jnp.mean(numbers_dead):0.2f}/{BATCH_SIZE}")
    print(f"mean_score={jnp.mean(mean_scores):0.2f}/epsilon={epsilon:0.4f}/lr={learning_rate:0.4f}")
    
    end_time = time.time()
    step_count = BATCH_SIZE*steps
    steps_per_second = step_count/(end_time - start_time)
    print(f"gen steps/s: {steps_per_second:0.2f}")
    
    # train on transition steps
    # replay buffer is guaranteed to be full
    # train on minibatch from replay buffer
    start_time = time.time()
    losses = jnp.zeros((grad_update_iterations,))
    for grad_update_iteration in range(grad_update_iterations):
        # randomly sample minibatch
        rolling_key, _ = jrand.split(rolling_key, 2)
        sample_size = BATCH_SIZE * replay_buffer_size
        sample_indices = jrand.choice(rolling_key, sample_size, shape=(minibatch_size,), replace=True)
        minibatch: ReplayBuffer = jax.tree_util.tree_map(lambda samples: samples[sample_indices], replay_buffer)
        # train on minibatch and get updated model params
        model_params, loss, grads = train_transition_minibatch(model_params, target_model_params, minibatch)
        # track losses
        losses = losses.at[grad_update_iteration].set(loss)


    print(f"train loop {train_loop_iteration}, avg_loss={jnp.mean(losses)}")
    end_time = time.time()
    train_steps = minibatch_size * grad_update_iterations
    train_samples_per_sec = train_steps / (end_time - start_time)
    print(f"train samples/sec: {train_samples_per_sec:0.2f}")

    print()



