import jax
from env import *
from dqn import *
from train_utils import *
from custom_types import *
import time

### OPTIONAL: TOGGLE DEBUG
DEBUG = False
if DEBUG:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)




## ------------------ ####
###  TRAIN LOOP SETUP  ###
#### ------------------ ##
rolling_key = jrand.PRNGKey(0)

### MODEL PARAMS
batch_size = 1024
sample_game_states: GameStateBatch = init_game_state_batch(rolling_key, batch_size)
input_size: int = get_model_vision_batched(sample_game_states).shape[-1]
output_size: int = 4 # up/dn/L/R
hidden_layers: int = 4
hidden_layer_size: int = 16
rolling_key, _ = jrand.split(rolling_key, 2)
model_params: MLPParams = init_mlp_dqn(rolling_key, input_size, output_size, hidden_layers, hidden_layer_size)

### RL PARAMS
epsilon: float = 1.0
epsilon_decay: float = 0.99
gamma: float = 0.99
learning_rate: float = 1e-4
learning_rate_decay: float = 0.99

### REPLAY BUFFER SETUP
replay_buffer_size: int = 512 # number of batches (steps), not number of samples
replay_buffer_index: int = 0

### TRAIN LOOP PARAMS
tau = 0.002 # interpolation constant for soft model updates
play_steps_per_iteration: int = 64 # autoregressive - how many batches to add to the replay buffer
grad_update_iterations: int = 128 # autoregressive - how many times to repeat minibatch sampling and training
minibatch_batches: int = 32 # batches
minibatch_size: int = minibatch_batches * batch_size # samples
train_loop_iterations: int = 1000000

### EVAL PARAMS
eval_steps = 16
eval_batch_size = 16 # batches



#### -------------------- ####
 ### TRAIN LOOP EXECUTION ###
#### -------------------- ####

### INITIALIZATION
# initialize game states
rolling_key, _ = jrand.split(rolling_key, 2) # reroll random key -> new key
game_state_batch = init_game_state_batch(rolling_key, batch_size)
# initialize target model
target_model_params = model_params
# initialize replay buffer
action_batch = take_action_batched(model_params, game_state_batch, rolling_key, epsilon=epsilon)
rolling_key, _ = jrand.split(rolling_key, 2)
transition_step_batch = update_game_state_batched(rolling_key, game_state_batch, action_batch)
replay_buffer: ReplayBuffer = init_replay_buffer(replay_buffer_size, transition_step_batch)

### RUN LOOP
for train_loop_iteration in range(train_loop_iterations):
    if train_loop_iteration == 0:
        trajectory_steps = replay_buffer_size # fill replay buffer first
    else:
        trajectory_steps = play_steps_per_iteration
    pass

    # putting everything in one giant jitted function looks ugly, sure
    # but it's a 100x speedup from the exact same code, pasted in this for-loop, non-jitted
    rolling_key, _ = jrand.split(rolling_key, 2) # reroll random key -> new key
    target_model_params, model_params, replay_buffer, game_state_batch, epsilon, learning_rate, mean_reward, mean_number_dead, mean_score, max_score, losses = run_train_loop_iteration(
        rolling_key,
        target_model_params,
        model_params,
        replay_buffer,
        game_state_batch,
        epsilon,
        learning_rate,
        grad_update_iterations,
        tau,
        epsilon_decay,
        learning_rate_decay,
        replay_buffer_size,
        eval_steps,
        eval_batch_size,
        trajectory_steps,
        minibatch_size,
        gamma,
        batch_size
    )

    print(f"generating: it={train_loop_iteration}/batches={trajectory_steps}/epsilon={epsilon:0.4f}/lr={learning_rate:0.6f}")
    print(f"eval: it={train_loop_iteration}/batches={eval_steps}/epsilon=0")
    ### PRINT STATS
    ## Eval stats
    print(f"mean_score={mean_score:0.2f}/"
          f"max_score={max_score}/"
          f"mean_reward={mean_reward:0.2f}/"
          f"deaths={mean_number_dead:0.2f}/{batch_size}/")
    ## Train stats
    #> note: avg loss == similarity to target model, not environment/game success
    print(f"avg_loss={jnp.mean(losses)}")
    print()


    



