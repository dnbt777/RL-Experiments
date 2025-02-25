from env import *
from dqn import *

from typing import NamedTuple, List, Optional
import time

# stores batches of transition steps
class ReplayBuffer(NamedTuple):
    state_batches: GameStateBatch     # GameStateBatch(buffer_size, batch_size, *)
    action_batches: jax.Array      # (buffer_size, batch_size,)
    reward_batches: jax.Array      # (buffer_size, batch_size,)
    next_state_batches: GameStateBatch  # GameStateBatch(buffer_size, batch_size)
    finished_batches: jax.Array          # (buffer_size, batch_size,)


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
        apples=jnp.zeros((buffer_size, batch_size, apple_count, 2)), #(buffer_size, batch_size, apple_count, 2)
        apple_count=jnp.zeros((buffer_size, batch_size)),
        snake_tail=jnp.zeros((buffer_size, batch_size, snake_tail_max_len)),
        snake_tail_length=jnp.zeros((buffer_size, batch_size)),
        snake_head=jnp.zeros((buffer_size, batch_size, 2))
    ),
    return ReplayBuffer(
        state_batches=init_game_state,
        action_batches=jnp.zeros((buffer_size, batch_size)),
        reward_batches=jnp.zeros((buffer_size, batch_size)),
        next_state_batches=init_game_state,
        finished_batches=jnp.zeros((buffer_size, batch_size)),
    )

# Append a TransitionStepBatch to the end of the replay buffer and override the first val with rolling
def append_replay_buffer(
        replay_buffer: ReplayBuffer,
        transition_step_batch: TransitionStepBatch
        ) -> ReplayBuffer:
    # roll replay buffer 1 to the left
    replay_buffer = jax.tree_util.tree_map(jnp.roll, replay_buffer) 
    # update the -1th index in the buffer
    buffer_update_fn = lambda buffer_arr, tsb_arr: buffer_arr.at[-1].set(tsb_arr)
    replay_buffer = jax.tree_util.tree_map(buffer_update_fn, replay_buffer, transition_step_batch)
    return replay_buffer
    

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
epsilon_decay: float = 1.0/10000
gamma: float = 0.1
# REPLAY BUFFER SETUP
replay_buffer_size: int = 1000
replay_buffer_index: int = 0
# TRAIN LOOP PARAMS
minibatch_size: int = 20
train_loop_iterations: int = 1000000
iterations_per_model_update: int = 2
play_steps_per_iteration: int = 20
grad_update_iterations: int = 500

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
        epsilon -= epsilon_decay
        epsilon = max(0, epsilon)
    # generate transition steps
    start_time = time.time()
    if train_loop_iteration == 0:
        steps = replay_buffer_size # fill replay buffer first
    else:
        steps = play_steps_per_iteration
    for step in range(steps):
        # Take an action step
        rolling_key, _ = jrand.split(rolling_key, 2)
        action_batch = take_action_batched(model_params, game_state_batch, rolling_key, epsilon=epsilon)
        # get next state
        # get action, reward(q)
        transition_step_batch = update_game_state_batched(game_state_batch, action_batch)
        # store in replay buffer
        if train_loop_iteration == 0 and step == 0:
            # init replay buffer by repeating transition steps
            replay_buffer: ReplayBuffer = init_replay_buffer(replay_buffer_size, transition_step_batch)
        else:
            replay_buffer = append_replay_buffer(replay_buffer, transition_step_batch) # roll and append to end
        replay_buffer_index = (replay_buffer_index + 1) % replay_buffer_size
        # update game state
        game_state_batch = transition_step_batch.next_state
        # replace finished game states with new ones
        fresh_game_state_batch = init_game_state_batch(rolling_key)
        game_state_batch = GameStateBatch(
            apples = jnp.where(
                transition_step_batch.finished[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.apples,
                game_state_batch.apples
            ),
            apple_count = jnp.where(
                transition_step_batch.finished[:],
                fresh_game_state_batch.apple_count,
                game_state_batch.apple_count
            ),
            snake_head = jnp.where(
                transition_step_batch.finished[:, jnp.newaxis],
                fresh_game_state_batch.snake_head,
                game_state_batch.snake_head
            ),
            snake_tail = jnp.where(
                transition_step_batch.finished[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.snake_tail,
                game_state_batch.snake_tail
            ),
            snake_tail_length = jnp.where(
                transition_step_batch.finished[:],
                fresh_game_state_batch.snake_tail_length,
                game_state_batch.snake_tail_length
            )
        ) # clunky. optimize manually, or jit so the compiler can attempt to optimize this
        mean_reward = jnp.mean(transition_step_batch.reward)
        episodes_finished = jnp.sum(transition_step_batch.finished)
        print(f"generated: buffsize={replay_buffer_index}/mean_reward={mean_reward:0.2f}/deaths={episodes_finished}/{BATCH_SIZE}")
    
    end_time = time.time()
    step_count = BATCH_SIZE*steps
    steps_per_second = step_count/(end_time - start_time)
    print(f"gen steps/s: {steps_per_second:0.2f}")
    
    # train on transition steps
    # replay buffer is guaranteed to be full
    for grad_update_iteration in range(grad_update_iterations):
        # train on minibatch from replay buffer
        rolling_key, _ = jrand.split(rolling_key, 2)
        minibatch_index = jrand.choice(rolling_key, replay_buffer_size, shape=(minibatch_size,), replace=True)
        #minibatch = jrand.choice(rolling_key, replay_buffer, shape=(minibatch_size,), replace=True, axis=0)

        # IMPORTANT todo: replace w vmap or proper batching
        for transition_step in minibatch:
            if transition_step is None:
                continue # make mypy happy
            # get target quality from bellman equation
            target_quality = jnp.max(get_action_qualities_batched(target_model_params, transition_step.next_state), axis=-1)
            target_Qsa = transition_step.reward + gamma * target_quality
            # get estimated quality of current state
            # backprop on target_Qsa == predicted_Qsa (MSE)
            loss_fn = lambda _model_params, _target_Qsa, _game_state: mse(_target_Qsa, jnp.max(get_action_qualities_batched(_model_params, _game_state)))
            loss, grads = jax.value_and_grad(loss_fn)(model_params, target_Qsa, transition_step.state)
            
        print("trained", train_loop_iteration, grad_update_iteration, transition_step.action, loss)


def train_transition_batch(model_params, target_model_params, minibatch):
    target_quality
