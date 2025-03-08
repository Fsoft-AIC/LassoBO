import numpy as np
import gym
from benchmark.filter import RunningStat


ENV_NAME = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2','Ant-v2']


class RLEnv:
    def __init__(self, env_name=ENV_NAME[0], seed=2025):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.seed=seed
        state_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        
        self.dims = state_dims * action_dims
        self.policy_shape = (action_dims, state_dims)
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.rs = RunningStat(state_dims)
        self.num_rollouts = 10
        self.max_time_steps = 1000
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        total_r = 0
        n_samples = 0
        for _ in range(self.num_rollouts):
            step = 0
            obs,_ = self.env.reset(seed = self.seed)
            while step < self.max_time_steps:
                self.rs.push(obs)
                norm_obs = (obs - self.rs.mean) / (self.rs.std + 1e-6)
                # norm_obs = (obs - self.mean)
                action = np.clip(np.dot(M, norm_obs),a_max=1., a_min=-1.)
                obs, r, done, _, _ = self.env.step(action)
                total_r += r
                n_samples += 1
                step += 1
                if done:
                    break
        
        return total_r / self.num_rollouts, n_samples / self.num_rollouts
    

if __name__ == '__main__':
    f = RLEnv()
    