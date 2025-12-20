import cv2
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from atariari.gym import gym
import numpy as np
import torch
from pathlib import Path
import os

# Import from stable-baselines3 instead of baselines
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import (
    AtariWrapper,
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.monitor import Monitor

# Import our standalone wrappers instead of a2c_ppo_acktr
from .a2c_wrappers import TimeLimitMask, TransposeImage, VecPyTorch, VecNormalize, VecPyTorchFrameStack
from .wrapper import AtariARIWrapper


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
    observation should be warped.
    """
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    """Stack k last frames."""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Handle gymnasium API (returns tuple)
        if isinstance(result, tuple):
            ob, info = result
        else:
            ob = result
            info = {}
        
        self.frames = []
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        result = self.env.step(action)
        # Handle both gym and gymnasium APIs
        if len(result) == 5:  # gymnasium
            ob, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # gym
            ob, reward, done, info = result
        
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0-1 range."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_atari(env_id, max_episode_steps=None):
    """Make Atari environment."""
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_env(env_id, seed, rank, log_dir, downsample=True, color=False):
    def _thunk():
        env = gym.make(env_id)

        # Check if this is an Atari environment (gymnasium with ale_py)
        # Atari envs typically have 'NoFrameskip' in their ID and have an 'ale' attribute
        is_atari = 'NoFrameskip' in env_id or hasattr(env.unwrapped, 'ale')
        if is_atari:
            env = make_atari(env_id)
            env = AtariARIWrapper(env)

        # Use gymnasium seed API
        env.reset(seed=seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)))

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env, downsample=downsample, color=color)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, num_frame_stack=1, downsample=True, color=False, gamma=0.99, log_dir='./tmp/', device=torch.device('cpu')):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    envs = [make_env(env_name, seed, i, log_dir, downsample, color)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs, start_method='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack > 1:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    return envs


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert observations to grayscale."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame


def wrap_deepmind(env, downsample=True, episode_life=True, clip_rewards=True, frame_stack=False, scale=False,
                  color=False):
    """Configure environment for DeepMind-style Atari."""
    if ("videopinball" in str(env.spec.id).lower()) or ('tennis' in str(env.spec.id).lower()) or ('skiing' in str(env.spec.id).lower()):
        env = WarpFrame(env, width=160, height=210, grayscale=False)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if downsample:
        env = WarpFrame(env, grayscale=False)
    if not color:
        env = GrayscaleWrapper(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env