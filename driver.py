import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import ray

import os
import csv
import time
import random
import datetime

from network import RL_Policy
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def writeToTensorBoard(writer, tensorboardData, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, returns, actorLoss, criticLoss, entropy, actorCriticGradNorm, travel_dist, success_rate, explored_rate = tensorboardData

    writer.add_scalar(tag='Losses/Actor Loss', scalar_value=actorLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Critic Loss', scalar_value=criticLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Actor Critic Grad Norm', scalar_value=actorCriticGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)

def main():
    # Handle devices for global training and local simulation
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize actor critic network
    actor_critic = RL_Policy(INPUT_DIM, 2).to(device)

    # Initialize optimizer
    actor_critic_optim = Adam(actor_critic.parameters(), lr=LR, eps=1e-5)

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        actor_critic.load_state_dict(checkpoint['policy_model'])
        actor_critic_optim.load_state_dict(checkpoint['policy_optimizer'])

        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(actor_critic_optim.state_dict()['param_groups'][0]['lr'])

        print(f"Learning... Running {MAX_TIMESTEP_PER_EPISODE} timesteps per episode, ", end='')

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    
    if device != local_device:
        actor_critic_weights = actor_critic.to(local_device).state_dict()
        actor_critic.to(device)
    else:
        actor_critic_weights = actor_critic.to(local_device).state_dict()

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(actor_critic_weights, curr_episode))
        # initialize metric collector

    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(5):
        experience_buffer.append([])

    try:
        while True:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(actor_critic_weights, curr_episode))

            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("Training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
                
                indices = range(len(experience_buffer[0]))
     
                # randomly sample a batch data
                sample_indices = random.sample(indices, BATCH_SIZE)
                rollouts = []
                for i in range(len(experience_buffer)):
                    rollouts.append([experience_buffer[i][index] for index in sample_indices])

                # Append episode data
                batch_obs = torch.stack(rollouts[0]).to(device)
                batch_acts = torch.stack(rollouts[1]).to(device)
                batch_log_probs = torch.stack(rollouts[2]).to(device)
                batch_rewards = torch.stack(rollouts[3]).to(device)
                batch_returns = torch.stack(rollouts[4]).to(device)

                # Calculate advantage
                curr_values = actor_critic.get_value(batch_obs)
                A_k = batch_returns - curr_values
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) #NOTE MIGHT NOT HAVE TO NORMALISE

                # training for n times each step
                for _ in range(N_UPDATES_PER_ITERATIONS):

                    # Calculate V_phi and pi_theta(a_t | s_t)
                    curr_values, curr_log_probs, dist_entropy = actor_critic.evaluate_actions(batch_obs, batch_acts)
    
                    # Calculate surrogate losses.
                    ratios = torch.exp(curr_log_probs - batch_log_probs)
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * A_k

                    # Calculate actor and critic losses.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(curr_values, batch_returns)

                    ''' Clipped Critic Loss'''
                    # value_pred_clipped = V_old.detach() + (V - V_old.detach()).clamp(-CLIP, CLIP)
                    # value_losses = (V - batch_returns).pow(2)
                    # value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                    # critic_loss = torch.max(value_losses, value_losses_clipped).mean()

                    actor_critic_loss = actor_loss\
                        + critic_loss * CRITIC_LOSS_COEF\
                        - dist_entropy * ENTROPY_COEF
                    
                    # print(f"actor l {actor_loss}")
                    # print(f"critic l {critic_loss}, {critic_loss * CRITIC_LOSS_COEF}")
                    # print(f"entropy l {dist_entropy}, {dist_entropy * ENTROPY_COEF}")
                    # print(f"total l {actor_critic_loss}")

                    # Calculate gradients and perform backward propagation for actor critic network
                    actor_critic_optim.zero_grad()
                    actor_critic_loss.backward()
                    actor_critic_grad_norm = nn.utils.clip_grad_norm_(actor_critic.parameters(),
                                            max_norm=MAX_GRAD_NORM, norm_type=2)
                    actor_critic_optim.step()

                    '''Check Gradients'''
                    # total_norm = 0
                    # for name, param in actor_critic.named_parameters():
                    #     if param.grad is not None:
                    #         print(f'Parameter: {name}, Gradient Norm: {param.grad.norm()}')
                    #         param_norm = param.grad.norm()
                    #         total_norm += param_norm.item() ** 2
                    # total_norm = total_norm ** (1. / 2)
                    # print(f"total norm {total_norm}")

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [batch_rewards.mean().item(), batch_returns.mean().item(), actor_loss.item(),
                        critic_loss.mean().item(), dist_entropy.mean().item(), actor_critic_grad_norm.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []
                    
            if device != local_device:
                actor_critic_weights = actor_critic.to(local_device).state_dict()
                actor_critic.to(device)
            else:
                actor_critic_weights = actor_critic.to(local_device).state_dict()
            
            # save the model
            if curr_episode % SAVE_FREQ == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": actor_critic.state_dict(),
                                "policy_optimizer": actor_critic_optim.state_dict(),
                                "episode": curr_episode,
                        }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                    

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    
if __name__ == "__main__":
    main()