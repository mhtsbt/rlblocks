from gym.spaces import Discrete
import gym

class ActionMapWrapper(gym.ActionWrapper):

    def __init__(self, env, action_map):
        self.action_map = action_map
        env.action_space = Discrete(len(action_map))

        super(ActionMapWrapper, self).__init__(env)

    def action(self, action):
        return self.action_map[action]
