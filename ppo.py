import torch
from torch import nn

from network import Global_Policy

max_timesteps_per_episode = 128
timesteps_per_batch = max_timesteps_per_episode * 128
gamma = 0.99


class PPO ():
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0] # 8 x g x g
        self.act_dim = env.action_space.shape[0] # do our selves

        self.actor = Global_Policy(self.obs_dim) # obs dimension and action post process
        self.critic = Global_Policy(self.obs_dim)

        def learn(self, total_timesteps):
            t_so_far = 0 # Timesteps simulated so far

            while t_so_far < total_timesteps:
                batch_obs, batch_acts, batch_log_probs, batch_returns, batch_episode_len = self.rollout()

    def rollout(self):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_returns = []
        batch_episode_len = []

        t = 0

        while t < self.timesteps_per_batch:
            epi_reward = [] 
            obs = self.env.reset() # TODO implement this basically inital obs
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs) # TODO implement
                obs, reward, done = self.env.step(action) # TODO implement

                epi_reward.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_episode_len.append(ep_t + 1) # +1 cos time step start from 0
            batch_rewards.append(epi_reward)
        
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        batch_returns = self.compute_returns(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_returns, batch_episode_len

    def compute_returns(self,batch_rewards):
        # The returns per episode per batch to return.
		# The shape will be (num timesteps per episode)
        batch_returns = []

		# Iterate through each episode
        for ep_rews in reversed(batch_rewards):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * gamma
                batch_returns.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_returns = torch.tensor(batch_returns, dtype=torch.float)

        return batch_returns
