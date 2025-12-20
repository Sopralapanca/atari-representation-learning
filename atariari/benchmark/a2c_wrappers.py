"""
Standalone implementations of a2c_ppo_acktr.envs wrappers
This file replaces the dependency on pytorch-a2c-ppo-acktr-gail
Save this as: atariari/benchmark/a2c_wrappers.py
"""

import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


class TimeLimitMask(gym.Wrapper):
    """
    Wrapper that adds 'bad_transition' flag to info dict when episode ends due to time limit.
    Converts gymnasium's 5-value returns to 4-value returns for compatibility with stable-baselines3.
    """
    def step(self, action):
        result = self.env.step(action)
        # Handle both gym and gymnasium APIs
        if len(result) == 5:  # gymnasium: obs, reward, terminated, truncated, info
            obs, rew, terminated, truncated, info = result
            done = terminated or truncated
            if truncated and hasattr(self.env, '_max_episode_steps'):
                if self.env._max_episode_steps == self.env._elapsed_steps:
                    info['bad_transition'] = True
        else:  # gym: obs, reward, done, info
            obs, rew, done, info = result
            if done and hasattr(self.env, '_max_episode_steps'):
                if self.env._max_episode_steps == self.env._elapsed_steps:
                    info['bad_transition'] = True
        # Always return 4 values for stable-baselines3 compatibility
        return obs, rew, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Handle both gym and gymnasium APIs
        if isinstance(result, tuple):
            return result[0]  # Return just observation for compatibility
        return result


class TransposeImage(gym.ObservationWrapper):
    """
    Transpose observation to PyTorch format (C, H, W).
    """
    def __init__(self, env, op=[2, 0, 1]):
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Transpose operation must be 3-dimensional"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    """
    Vectorized environment wrapper that converts numpy arrays to PyTorch tensors.
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment wrapper that normalizes observations and optionally rewards.
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, norm_reward=True):
        VecEnvWrapper.__init__(self, venv)
        
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        self.norm_obs = ob
        self.norm_reward = norm_reward

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        
        if self.training and self.norm_obs:
            self.ob_rms.update(obs)
        obs = self._obfilt(obs)
        
        if self.norm_reward:
            self.ret = self.ret * self.gamma + rews
            if self.training and self.ret_rms:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), 
                          -self.cliprew, self.cliprew)
        
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                         -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        if self.norm_obs:
            return self._obfilt(obs)
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class RunningMeanStd(object):
    """
    Tracks running mean and std of observations.
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class VecPyTorchFrameStack(VecEnvWrapper):
    """
    Vectorized environment wrapper that stacks multiple frames.
    """
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()