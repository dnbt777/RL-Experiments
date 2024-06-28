
from config import *

def initialize_q_table():
    return {}

def get_q_value(Q, state, action):
    return Q.get((state, action), 0.0)

def update_q_value(Q, state, action, reward, next_state):
    old_value = get_q_value(Q, state, action)
    next_max = max(get_q_value(Q, next_state, a) for a in range(4))
    new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
    Q[(state, action)] = new_value
    return new_value
