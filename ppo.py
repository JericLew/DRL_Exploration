import torch
from torch import nn
from torch.optim import Adam
import numpy as np

import datetime
import time
import csv

from worker import Worker
from network import RL_Policy
from parameter import *

class PPO ():
    def __init__(self):
        self.device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
        self.local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

        # initialize neural networks
        self.actor_critic = RL_Policy(INPUT_DIM, 2).to(self.device)

        # Initialize optimizers
        self.actor_critic_optim = Adam(self.actor_critic.parameters(), lr=LR, eps=1e-5)

        # init csv
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')  # Get the current timestamp
        self.csv_file = f'iteration_data_{timestamp}.csv'
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Iteration", 
                "Average Episodic Length",
                "Average Episodic Return",
                "Average Actor Loss", 
                "Average Critic Loss",                 
                "Timesteps So Far",
                "Iteration Time (secs)",
            ])

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
        }
            
    def learn(self, total_timesteps):
        print(f"Learning... Running {MAX_TIMESTEP_PER_EPISODE} timesteps per episode, ", end='')
        print(f"{TIMESTEP_PER_BATCH} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        epi_so_far = 0
        i_so_far = 0 # Iterations ran so far

        while t_so_far < total_timesteps:

            batch_obs, batch_acts, batch_returns, batch_episode_len, epi_so_far = self.rollout(epi_so_far)
            
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_episode_len)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            
            # Calculate advantage at k-th iteration
            V, batch_log_probs, dist_entropy = self.actor_critic.evaluate(batch_obs, batch_acts)
            A_k = batch_returns - V.detach()   
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) #NOTE MIGHT NOT HAVE TO NORMALISE

            print(A_k)
            # torch.nn.utils.grad_norm to check grad NOTE

            # print(f"V iter {V.size()}")
            # print(f"A_k iter {A_k.size()}")
            # print(f"log_prob batch {batch_log_probs}")
            # print(f"batch_returns iter {batch_returns.size()}")
            
            for _ in range(N_UPDATES_PER_ITERATIONS):                                                       # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, dist_entropy = self.actor_critic.evaluate(batch_obs, batch_acts)

                # Calculate surrogate losses.
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * A_k

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_returns)


                actor_critic_loss = actor_loss
                # actor_critic_loss = critic_loss * CRITIC_LOSS_COEF +\
                #     actor_loss - dist_entropy * ENTROPY_COEF
                
                # print(f"curr_log_probs grad: {curr_log_probs.requires_grad}")
                # print(f"batch_log_probs grad: {batch_log_probs.requires_grad}")
                # print(f"V grad: {V.requires_grad}")
                # print(f"A_k grad: {A_k.requires_grad}")
                # print(f"ratios grad: {ratios.requires_grad}")
                # print(f"actor loss grad: {actor_loss.requires_grad}")
                # print(f"critic loss grad: {critic_loss.requires_grad}")
                # print(f"dist entropy grad: {dist_entropy.requires_grad}")
                # print(f"actor critic loss grad: {actor_critic_loss.requires_grad}")
                
                # Calculate gradients and perform backward propagation for actor network
                self.actor_critic_optim.zero_grad()
                actor_critic_loss.backward(retain_graph=True)
                self.actor_critic_optim.step()

                # Check gradients
                for name, param in self.actor_critic.named_parameters():
                    if param.grad is not None:
                        print(f'Parameter: {name}, Gradient Norm: {param.grad.norm()}')
                
                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % SAVE_FREQ == 0:
                torch.save(self.actor_critic.state_dict(), './ppo_actor_critic.pth')

    def rollout(self, epi_so_far):

        if self.device != self.local_device:
            actor_critic_weights = self.actor_critic.to(self.local_device).state_dict()
            self.actor_critic.to(self.device)
        else:
            actor_critic_weights = self.actor_critic.to(self.local_device).state_dict()
        
        # batch data
        batch_obs = []
        batch_acts = []
        batch_rewards = []
        batch_returns = []
        batch_episode_len = []

        for _ in range(EPISODE_PER_BATCH):
            save_img = False if epi_so_far % SAVE_IMG_GAP == 0 else False

            worker = Worker(epi_so_far, actor_critic_weights, save_image=save_img)
            worker.work(epi_so_far)
            epi_so_far += 1

            with torch.no_grad():  # Apply no_grad to optimize memory usage
                batch_obs.append(torch.stack(worker.episode_obs))
                batch_acts.append(torch.stack(worker.episode_acts))
                batch_rewards.append(worker.episode_rewards)
                batch_episode_len.append(worker.episode_len)
        
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.cat(batch_obs).to(self.device)
        batch_acts = torch.cat(batch_acts).to(self.device)
        batch_returns = self.compute_returns(batch_rewards)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_episode_len

        return batch_obs, batch_acts, batch_returns, batch_episode_len, epi_so_far

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
                discounted_reward = rew + discounted_reward * GAMMA
                batch_returns.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        # with torch.no_grad():
        batch_returns = torch.tensor(batch_returns, dtype=torch.float).to(self.device)

        return batch_returns
                   
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
        avg_actor_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['critic_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 10))
        avg_critic_loss = str(round(avg_critic_loss, 10))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # i_so_far starts from 1 in a way
        def write_iteration_data(i_so_far, avg_ep_lens, avg_ep_rews, avg_actor_loss, avg_critic_loss,t_so_far, delta_t):
            data = [i_so_far, avg_ep_lens, avg_ep_rews, avg_actor_loss, avg_critic_loss, t_so_far, delta_t]
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)

        write_iteration_data(i_so_far, avg_ep_lens, avg_ep_rews, avg_actor_loss, avg_critic_loss, t_so_far, delta_t)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []