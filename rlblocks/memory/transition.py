from collections import namedtuple

Transition = namedtuple('Transition', ['start_state', 'result_state', 'action', 'reward', 'done', 'timestep'])