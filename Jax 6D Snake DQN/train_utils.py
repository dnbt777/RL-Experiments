from env import *
from dqn import *
from custom_types import *


#### --------####------####-------- ####
## DATA INITIALIZATION AND CONVERSION ##
## ---- #### ---- #### ---- #### ---- ##

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
        batch_size: int,
        ) -> ReplayBuffer:
    # roll replay buffer batch_size to the left
    replay_buffer = jax.tree_util.tree_map(lambda buffer: jnp.roll(buffer, -batch_size, axis=0), replay_buffer) 
    # update the end of the buffer to the latest batch of values
    buffer_update_fn = lambda buffer_arr, tsb_arr: buffer_arr.at[-batch_size:].set(tsb_arr)
    transition_step_batch_as_buffer = tsb_to_replay_buffer(transition_step_batch)
    replay_buffer = jax.tree_util.tree_map(buffer_update_fn, replay_buffer, transition_step_batch_as_buffer)
    return replay_buffer


# Runs at the start of each iteration
# Updates target model and params
@functools.partial(jax.jit, static_argnames=["epsilon_decay", "tau", "learning_rate_decay"])
def update_train_loop(
        target_model_params: MLPParams,
        model_params: MLPParams,
        epsilon: float,
        learning_rate: float,
        tau: float,
        epsilon_decay: float,
        learning_rate_decay: float
        ) -> Tuple[MLPParams, float, float]:
    target_model_params = jax.tree_util.tree_map(lambda target_param, param: (1 - tau)*target_param + tau*param, target_model_params, model_params)
    # decay epsilon
    epsilon = epsilon*epsilon_decay
    learning_rate = learning_rate*learning_rate_decay
    return target_model_params, epsilon, learning_rate



@functools.partial(jax.jit, static_argnames=["eval_steps", "eval_batch_size"])
def model_eval(
        model_params: MLPParams,
        rolling_key: jax.Array,
        eval_steps: int,
        eval_batch_size: int,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # init game state batch
    eval_game_state_batch = init_game_state_batch(rolling_key, eval_batch_size)
    # define scan function
    def scanf(game_state_batch, rolling_scan_key):
        # Take an action step
        action_batch = take_action_batched(model_params, game_state_batch, rolling_scan_key, epsilon=0)
        # get next state
        # get action, reward(q)
        rolling_scan_key, _ = jrand.split(rolling_scan_key, 2)
        transition_step_batch: TransitionStepBatch = update_game_state_batched(rolling_scan_key, game_state_batch, action_batch)
        # update game state
        next_game_state_batch = transition_step_batch.next_state_batch
        # replace finished game states with new ones, conditionally (no branching == better compilation)
        rolling_scan_key, _ = jrand.split(rolling_scan_key, 2)
        batch_size = transition_step_batch.finished_batch.shape[0]
        fresh_game_state_batch = init_game_state_batch(rolling_scan_key, batch_size)
        next_game_state_batch = GameStateBatch(
            apples = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.apples,
                next_game_state_batch.apples
            ),
            score = jnp.where(
                transition_step_batch.finished_batch[:],
                fresh_game_state_batch.score,
                next_game_state_batch.score
            ),
            snake_head = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis],
                fresh_game_state_batch.snake_head,
                next_game_state_batch.snake_head
            ),
            snake_tail = jnp.where(
                transition_step_batch.finished_batch[:, jnp.newaxis, jnp.newaxis],
                fresh_game_state_batch.snake_tail,
                next_game_state_batch.snake_tail
            ),
            snake_tail_length = jnp.where(
                transition_step_batch.finished_batch[:],
                fresh_game_state_batch.snake_tail_length,
                next_game_state_batch.snake_tail_length
            )
        ) # look into jax.lax.cond for potential speedup.
        # return statistics_i => b
        mean_reward = jnp.mean(transition_step_batch.reward_batch)
        mean_score = jnp.mean(transition_step_batch.state_batch.score)
        number_dead = jnp.sum(transition_step_batch.finished_batch)
        max_score = jnp.max(transition_step_batch.state_batch.score)
        return next_game_state_batch, (mean_reward, mean_score, number_dead, max_score)
    
    # do the scan using the scan function
    scan_keys = jrand.split(rolling_key, eval_steps)
    end_state, b = jax.lax.scan(scanf, eval_game_state_batch, scan_keys) # b is (n, 4)

    # extract the data from the scan
    mean_rewards, mean_scores, numbers_dead, max_scores = b # (4, n)
    mean_reward = jnp.mean(mean_rewards)
    mean_number_dead = jnp.mean(numbers_dead)
    mean_score = jnp.mean(mean_scores)
    max_score = jnp.max(max_scores)
    
    return mean_reward, mean_number_dead, mean_score, max_score



###~~~##############   
## MODEL TRAINING ##
####################

@jax.jit
def train_transition_minibatch(
        model_params: MLPParams,
        target_model_params: MLPParams,
        minibatch: ReplayBuffer,
        learning_rate: float,
        gamma: float,
        ) -> Tuple[MLPParams, float, MLPParams]:
    target_next_quality = jnp.max(get_action_qualities_batched(target_model_params, minibatch.next_state_batch), axis=-1)
    target_Qsa = minibatch.reward_batch + gamma*target_next_quality*(1 - minibatch.finished_batch) # bellman equation term
    # get estimated quality of current state,
    # then backprop on target_Qsa == predicted_Qsa (MSE)
    # defining a function in-function like this pre-grad or value_and_grad seems to be standard for jax code. it gets jitted anyways
    def loss_fn(_model_params, _target_Qsa, _game_state_batch, _action_batch):
        action_qualities = get_action_qualities_batched(_model_params, _game_state_batch) # predicted Q(s, a)  (batch, action, quality)
        actual_action = _action_batch # (batch, action)
        batch_size = _action_batch.shape[0]
        predicted_Qsa = jnp.take_along_axis(action_qualities, actual_action[:, None].astype(jnp.int32), axis=-1).squeeze(-1)
        return mse(_target_Qsa, predicted_Qsa)
    loss, grads = jax.value_and_grad(loss_fn)(model_params, target_Qsa, minibatch.state_batch, minibatch.action_batch)
    # update model
    model_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate*g, model_params, grads)
    return model_params, loss, grads


# samples minibatches from the replay buffer and trains/updates the model on them
@functools.partial(jax.jit, static_argnames=["gamma", "grad_update_iterations", "minibatch_size", "batch_size", "replay_buffer_size"])
def train_transition_minibatches(
    rolling_key: jax.Array,
    model_params: MLPParams,
    target_model_params: MLPParams,
    replay_buffer: ReplayBuffer,
    grad_update_iterations: int,
    minibatch_size: int,
    learning_rate: float,
    gamma: float,
    batch_size: int,
    replay_buffer_size: int
    ) -> Tuple[MLPParams, jax.Array]:
    # train on transition steps
    # train on minibatch from replay buffer
    population_size = batch_size*replay_buffer_size

    ## Set up the scan function
    def scanf(model_params, scan_key):
        # randomly sample minibatch from replay buffer (replay buffer is guaranteed to be full)
        # replay buffer: (batch, *)
        # get n samples from the batch dim
        # total samples in replay buffer = replay_buffer_batches*batch_size
        sample_indices = jrand.choice(scan_key, population_size, shape=(minibatch_size,), replace=True)
        minibatch: ReplayBuffer = jax.tree_util.tree_map(lambda samples: samples[sample_indices], replay_buffer)
        # train on minibatch and get updated model params
        model_params, loss, grads = train_transition_minibatch(model_params, target_model_params, minibatch, learning_rate, gamma)
        return model_params, loss

    scan_keys = jrand.split(rolling_key, grad_update_iterations)
    model_params, losses = jax.lax.scan(scanf, model_params, scan_keys)

    return model_params, losses



    ##############
############# ########
## GENERATE REPLAYS ##
#### #################
    ##############

@jax.jit
def generate_replays(
        replay_buffer: ReplayBuffer,
        model_params: MLPParams,
        game_state_batch: GameStateBatch,
        rolling_key: jax.Array,
        epsilon: float,
        ) -> Tuple[ReplayBuffer, GameStateBatch]:
    # jitting this for-loop will make compilation time slower and increase memory
    # but this is better than not jitting this code, which is bottlenecking training
    # TODO: turn the for-loop into a scan
    # Take an action step
    rolling_key, _ = jrand.split(rolling_key, 2)
    action_batch = take_action_batched(model_params, game_state_batch, rolling_key, epsilon)
    # get next state
    # get action, reward(q)
    rolling_key, _ = jrand.split(rolling_key, 2)
    transition_step_batch: TransitionStepBatch = update_game_state_batched(rolling_key, game_state_batch, action_batch)
    # store in replay buffer
    batch_size = transition_step_batch.finished_batch.shape[0]
    replay_buffer = append_replay_buffer(replay_buffer, transition_step_batch, batch_size) # roll and append to end
    # update game state
    game_state_batch = transition_step_batch.next_state_batch
    # replace finished game states with new ones
    rolling_key, _ = jrand.split(rolling_key, 2)
    fresh_game_state_batch = init_game_state_batch(rolling_key, batch_size)
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
    )
    return replay_buffer, game_state_batch


# Given a number of steps, will generate trajectories of length {steps}
# and add them to the replay buffer
@functools.partial(jax.jit, static_argnames=["steps"])
def generate_replay_trajectories(
        replay_buffer: ReplayBuffer,
        model_params: MLPParams,
        game_state_batch: GameStateBatch,
        rolling_key: jax.Array,
        epsilon: float,
        steps: int
    ) -> Tuple[ReplayBuffer, GameStateBatch]:

    def scanf(state, scan_key):
        replay_buffer, game_state_batch = state
        replay_buffer, game_state_batch = generate_replays(
            replay_buffer,
            model_params,
            game_state_batch,
            scan_key,
            epsilon
        )
        return (replay_buffer, game_state_batch), None

    scan_keys = jrand.split(rolling_key, steps)
    initial_state = (replay_buffer, game_state_batch)
    final_state, _ = jax.lax.scan(scanf, initial_state, scan_keys)
    replay_buffer, game_state_batch = final_state
    return replay_buffer, game_state_batch



########################################################
##                                                    ##
##   888b     d888        d8888 8888888 888b    888   ##
##   8888b   d8888       d88888   888   8888b   888   ##
##   88888b.d88888      d88P888   888   88888b  888   ##
##   888Y88888P888     d88P 888   888   888Y88b 888   ##
##   888 Y888P 888    d88P  888   888   888 Y88b888   ##
##   888  Y8P  888   d88P   888   888   888  Y88888   ##
##   888   V   888  d8888888888   888   888   Y8888   ##
##   888       888 d88P     888 8888888 888    Y888   ##
##                                                    ##
########################################################                                 

@functools.partial(jax.jit, static_argnames=[
    "grad_update_iterations", "tau", "epsilon_decay", "learning_rate_decay",
    "replay_buffer_size", "eval_steps", "eval_batch_size", "trajectory_steps",
    "minibatch_size", "gamma", "batch_size"])
def run_train_loop_iteration(
        rolling_key: jax.Array,
        target_model_params: MLPParams,
        model_params: MLPParams,
        replay_buffer: ReplayBuffer,
        game_state_batch: GameStateBatch,
        epsilon: float,
        learning_rate: float,
        grad_update_iterations: int,
        tau: float,
        epsilon_decay: float,
        learning_rate_decay: float,
        replay_buffer_size: int,
        eval_steps: int,
        eval_batch_size: int,
        trajectory_steps: int,
        minibatch_size: int,
        gamma: float,
        batch_size: int
        ) -> Tuple[MLPParams, MLPParams, ReplayBuffer, GameStateBatch, float, float, jax.Array, jax.Array, jax.Array, jax.Array, float]:
    ### UPDATE PARAMS/TARGET MODEL
    target_model_params, epsilon, learning_rate = update_train_loop(
        target_model_params, model_params, epsilon, learning_rate, tau, epsilon_decay, learning_rate_decay
    ) 
    ### GENERATE AND STORE STEPS
    rolling_key, _ = jrand.split(rolling_key, 2)
    replay_buffer, game_state_batch = generate_replay_trajectories(
        replay_buffer,
        model_params,
        game_state_batch,
        rolling_key,
        epsilon,
        trajectory_steps
    )
    ### TRAIN ON STORED STEPS
    #> sample minibatches from the replay buffer
    #> then train+update the model on them
    rolling_key, _ = jrand.split(rolling_key, 2)
    model_params, losses = train_transition_minibatches(
        rolling_key,
        model_params,
        target_model_params,
        replay_buffer,
        grad_update_iterations,
        minibatch_size,
        learning_rate,
        gamma,
        batch_size,
        replay_buffer_size
    )
    ### EVALUATE MODEL (epsilon=0)
    rolling_key, _ = jrand.split(rolling_key, 2)
    mean_reward, mean_number_dead, mean_score, max_score = model_eval(
        model_params,
        rolling_key,
        eval_steps,
        eval_batch_size,
    )
    return target_model_params, model_params, replay_buffer, game_state_batch, epsilon, learning_rate, mean_reward, mean_number_dead, mean_score, max_score, losses