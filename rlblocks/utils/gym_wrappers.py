from gym.spaces import Discrete, Box
from copy import deepcopy
from collections import deque
from rlblocks.utils.img_preprocessing import ImgPreprocessing
import numpy as np
import gym


class ActionMapWrapper(gym.ActionWrapper):

    def __init__(self, env, action_map):
        self.action_map = action_map
        env.action_space = Discrete(len(action_map))

        super(ActionMapWrapper, self).__init__(env)

    def action(self, action):
        return self.action_map[action]

class FrameSkipWrapper(gym.Wrapper):

    def __init__(self, env, fs):
        self.fs = fs
        super(FrameSkipWrapper, self).__init__(env)

    def step(self, action):

        step_reward = 0

        for _ in range(self.fs):
            state, reward, done, info = self.env.step(action=action)
            step_reward += reward

            if done:
                break

        return state, step_reward, done, info


class FrameHistoryWrapper(gym.ObservationWrapper):

    def __init__(self, env, n_frames, frame_w, frame_h):
        super(FrameHistoryWrapper, self).__init__(env)
        self.buffer = deque(maxlen=n_frames)
        self.n_frames = n_frames

        # fill buffer with empty frames
        for _ in range(self.n_frames):
            self.buffer.append(
                np.zeros(shape=(frame_w, frame_h), dtype=np.int8))

    def observation(self, observation):
        self.buffer.append(observation)
        return deepcopy(self.buffer)


class ResizeGreyscaleWrapper(gym.ObservationWrapper):

    def __init__(self, env, frame_w=84, frame_h=84):
        super(ResizeGreyscaleWrapper, self).__init__(env)
        self.frame_w = frame_w
        self.frame_h = frame_h

        # change the observation-space to the new format
        self.observation_space = Box(low=0, high=255, shape=(self.frame_h, self.frame_w, 1))

    def observation(self, observation):
        observation = ImgPreprocessing.greyscale(observation)
        observation = ImgPreprocessing.resize(
            observation, width=self.frame_w, height=self.frame_h)
        return observation
