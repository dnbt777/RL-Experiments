from env import *
from dqn import *

from typing import NamedTuple, List, Optional


class TransitionStep(NamedTuple):
    state: GameState
    action: int
    reward: float
    next_state: GameState
    finished: bool



## TRAIN LOOP SETUP
rolling_key = jrand.PRNGKey(0)
# MODEL PARAMS
sample_game_state: GameState = init_game_state(rolling_key)
input_size: int = len(get_model_vision(sample_game_state))
output_size: int = 4 # up/dn/L/R
hidden_layers: int = 5
hidden_layer_size: int = 16
rolling_key, _ = jrand.split(rolling_key, 2)
model_params: MLPParams = init_mlp_dqn(rolling_key, input_size, output_size, hidden_layers, hidden_layer_size)
# REPLAY BUFFER SETUP
replay_buffer_size: int = 100
replay_buffer_index: int = 0
replay_buffer: List[Optional[TransitionStep]] = [None for _ in range(replay_buffer_size)] # contains a list of either TransitionStep or None
replay_buffer_is_full: bool = False
minibatch_size: int = 50
# TRAIN LOOP PARAMS
episodes: int = 1000000
max_steps_per_episode: int = 100
epsilon: float = 1.0
epsilon_decay: float = 1.0/10000
gamma: float = 0.1

## TRAIN LOOP EXECUTION
loss: float = 0
episode_returns: List[float] = []
for episode in range(episodes):
    if episode % 100 == 0:
        target_model_params = model_params
    # Reinit game state
    rolling_key, _ = jrand.split(rolling_key, 2) # reroll random key -> new key
    game_state = init_game_state(rolling_key)
    epsilon -= epsilon_decay
    epsilon = max(0, epsilon)
    for step in range(max_steps_per_episode):
        # Take an action step
        rolling_key, _ = jrand.split(rolling_key, 2)
        action = take_action(model_params, game_state, rolling_key, epsilon=epsilon)
        # get next state
        # get action, reward(q)
        next_game_state, reward, finished = update_game_state(game_state, action)
        # store in replay buffer
            # replay_buffer[end_index] = (action, reward)
            # end_index = (end_index + 1) % replay_buffer_size
        transition_step = TransitionStep(
            state = game_state,
            action = action,
            reward = reward,
            next_state = next_game_state,
            finished = finished
        )
        replay_buffer[replay_buffer_index] = transition_step
        replay_buffer_index = (replay_buffer_index + 1) % replay_buffer_size

        game_state = next_game_state

        if finished and not replay_buffer_is_full:
            print(f"generated: ep={episode}/step={step}/action={action}/buffsize={replay_buffer_index}")
    
    # train once per episode generation
    if replay_buffer_is_full or (not (None in replay_buffer)):
        # once buffer is full, this stops checking if 'None' is in it
        replay_buffer_is_full = True
        # train on minibatch from replay buffer
        minibatch: List[TransitionStep] = replay_buffer[-minibatch_size:] # TODO randomize sampling

        # IMPORTANT todo: replace w vmap or proper batching
        for transition_step in minibatch:
            if transition_step is None:
                continue # make mypy happy
            # get target quality from bellman equation
            target_quality = jnp.max(get_action_qualities(target_model_params, transition_step.next_state), axis=-1)
            target_Qsa = transition_step.reward + gamma * target_quality
            # get estimated quality of current state
            # backprop on target_Qsa == predicted_Qsa (MSE)
            loss_fn = lambda _model_params, _target_Qsa, _game_state: mse(_target_Qsa, jnp.max(get_action_qualities(_model_params, _game_state)))
            loss, grads = jax.value_and_grad(loss_fn)(model_params, target_Qsa, transition_step.state)
            
        episode_returns.append(reward) # will become large. todo optimize or remove
        print("trained", episode, step, transition_step.action, loss)


def train_transition_batch(model_params, target_model_params, minibatch):
    target_quality = jnp.max(get_action_qualities(target_model_params))
