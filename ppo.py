import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import time
from worker import Worker
from network import Global_Policy
from utils import DiagGaussian


class PPO ():
    def __init__(self, network, **hyperparameters):
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.actor = network((8,240,320), hidden_size=256) # obs dimension and action post process
        self.critic = network((8,240,320), hidden_size=256)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.dist = DiagGaussian(self.actor.output_size, 2) # 256, 2

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }
            
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        epi_so_far = 0
        i_so_far = 0 # Iterations ran so far

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_returns, batch_episode_len, epi_so_far = self.rollout(epi_so_far)
            
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_episode_len)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_returns - V.detach()   
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_returns)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')


    def rollout(self, epi_so_far):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_returns = []
        batch_episode_len = []

        for _ in range(self.episode_per_batch): # TODO 128 *2 = 256 , timestep per batch
            worker = Worker(epi_so_far, self.actor,\
                            max_timestep = self.max_timesteps_per_episode,save_image=False)
            worker.work(epi_so_far)
            epi_so_far += 1

            batch_obs.append(torch.stack(worker.episode_obs))
            batch_acts.append(torch.stack(worker.episode_acts))
            batch_log_probs.append(torch.stack(worker.episode_log_probs))
            batch_rewards.append(worker.episode_rewards)
            batch_episode_len.append(worker.episode_len)

        # t = 0

        # while t < self.timesteps_per_batch:
        #     epi_reward = [] 
        #     obs = self.env.reset() # TODO implement this basically inital obs
        #     done = False

        #     for ep_t in range(self.max_timesteps_per_episode):
        #         t += 1

        #         batch_obs.append(obs)

        #         action, log_prob = self.get_action(obs) # TODO implement
        #         obs, reward, done = self.env.step(action) # TODO implement

        #         epi_reward.append(reward)
        #         batch_acts.append(action)
        #         batch_log_probs.append(log_prob)

        #         if done:
        #             break

        #     batch_episode_len.append(ep_t + 1) # +1 cos time step start from 0
        #     batch_rewards.append(epi_reward)
        
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.cat(batch_obs)
        batch_acts = torch.cat(batch_acts)
        batch_log_probs = torch.cat(batch_log_probs)
        batch_returns = self.compute_returns(batch_rewards)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_episode_len

        return batch_obs, batch_acts, batch_log_probs, batch_returns, batch_episode_len, epi_so_far

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
                discounted_reward = rew + discounted_reward * self.gamma
                batch_returns.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_returns = torch.tensor(batch_returns, dtype=torch.float)

        return batch_returns

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V, _ = self.critic(batch_obs)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        _, actor_features = self.actor(batch_obs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(batch_acts)
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, action_log_probs
    
    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.episode_per_batch = 2
        self.max_timesteps_per_episode = 32           # Max number of timesteps per episode
        self.timesteps_per_batch =  self.episode_per_batch *\
            self.max_timesteps_per_episode               # Number of timesteps to run per batch
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")
               
    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

model = PPO(Global_Policy)
model.learn(1280)