import copy
import os
import matplotlib.pyplot as plt

import imageio
import numpy as np
import torch
import torch.nn as nn

from env import Env
from parameter import *

from network import Global_Policy 
from utils import DiagGaussian

class Worker:
    def __init__(self, global_step, weights, save_image=False):
        self.device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
        self.local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

        self.max_timestep = MAX_TIMESTEP_PER_EPISODE
        self.k_size = K_SIZE
        
        self.global_step = global_step
        self.actor = Global_Policy(INPUT_DIM, hidden_size=HIDDEN_SIZE).to(self.local_device)
        self.actor.load_state_dict(weights)

        # Initialize distribution
        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(2,), fill_value=0.5).to(self.local_device)
        self.cov_mat = torch.diag(self.cov_var).to(self.local_device)
        self.dist = torch.distributions.MultivariateNormal
        # self.dist = DiagGaussian(self.actor.output_size, 2).to(self.local_device) # 256, 2

        self.save_image = save_image
        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)

        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_obs = []
        self.episode_acts = []
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_len = []

    def get_local_map_boundaries(self, robot_position, local_size, full_size):
        x_center, y_center = robot_position
        local_h, local_w = local_size
        full_h, full_w = full_size
        x_start, y_start = x_center - local_w // 2, y_center - local_h // 2
        x_end, y_end = x_start + local_w, y_start + local_h

        if x_start < 0:
            x_start, x_end = 0, local_w
        if x_end >= full_w:
            x_start, x_end = full_w - local_w, full_w
        if y_start < 0:
            y_start, y_end = 0, local_h
        if y_end >= full_h:
            y_start, y_end = full_h - local_h, full_h

        local_robot_y = y_center - y_start
        local_robot_x = x_center - x_start

        return y_start, y_end, x_start, x_end, local_robot_y, local_robot_x

    def get_observations(self): # get 8 x g x g input for model
        # observation[0, :, :] probability of obstacle
        # observation[1, :, :] probability of exploration
        # observation[2, :, :] indicator of current position
        # observation[3, :, :] indicator of visited

        # TODO make it less computationally intensive (update local only then patch on global)
        robot_belief = copy.deepcopy(self.env.robot_belief)
        visited_map = copy.deepcopy(self.env.visited_map)
        ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
        local_size = (int(ground_truth_size[0] / 2) ,int(ground_truth_size[1] / 2))  # (h,w) # TODO 2 is a downsize parameter
        
        global_map = torch.zeros(4, ground_truth_size[0], ground_truth_size[1]).to(self.local_device)
        local_map = torch.zeros(4, local_size[0], local_size[1]).to(self.local_device)
        observations = torch.zeros(8, local_size[0], local_size[1]).to(self.local_device) # (8,height,width)

        lmb = self.get_local_map_boundaries(self.robot_position, local_size, ground_truth_size)

        # Create a mask for each condition
        mask_obst = (robot_belief == 1) # if colour 1 : index 0 = 1, index 1 = 1 obst
        mask_free = (robot_belief == 255) # if colour 255: index 0 = 0, index 1 = 1 free
        mask_unkn = (robot_belief == 127) # if colour 127: index 0 = 0, index 1 = 0 unkw
        mask_visi = (visited_map == 1) # if visited: index : 3 = 1 vist

        # Update robot_belief based on the masks
        global_map[0, mask_obst] = 1
        global_map[1, mask_obst] = 1
        global_map[0, mask_free] = 0
        global_map[1, mask_free] = 1
        global_map[0, mask_unkn] = 0
        global_map[1, mask_unkn] = 0
        global_map[3, mask_visi] = 1
        global_map[2, self.robot_position[1] - 2:self.robot_position[1] + 3, \
                    self.robot_position[0] - 2:self.robot_position[0] + 3] = 1
                    # if robot_y and robot_x: index 2 = 1 
        
        local_map = global_map[:, lmb[0]:lmb[1], lmb[2]:lmb[3]] # (width,height)

        observations[0:4, :, :] = local_map.detach()
        observations[4:, :, :] = nn.MaxPool2d(2)(global_map)
        # TODO magic number (but is a pooling from algo)

        # # map check uncomment to check output of observation
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(robot_belief, cmap='gray')
        # axes[1].imshow(global_map[2, : :], cmap='gray') 
        # axes[2].imshow(local_map[2, : :], cmap='gray')
        # plt.savefig('output.png')
        return observations
    
    def act(self, observations, actor):
        with torch.no_grad():
            value, actor_features = actor(observations.unsqueeze(0)) #add batch dimension
            dist = self.dist(actor_features, self.cov_mat)
            action = dist.sample().squeeze() # squeeze because it was made for multibatch input
            action_log_probs = dist.log_prob(action).squeeze()
            # print(f"action {action}")
            # print(f"logprobs {action_log_probs}")
        return value, action.detach(), action_log_probs.detach()

    def save_observations(self, observations):
        self.episode_obs.append(observations)

    def save_action(self, action_features, action_log_probs):
        self.episode_acts.append(action_features)
        self.episode_log_probs.append(action_log_probs)

    def save_reward_done(self, reward, done):
        self.episode_rewards.append(reward)


    def find_target_pos(self, action_features):
        with torch.no_grad():
                # process actor output to target_position
                post_sig_action = nn.Sigmoid()(action_features).cpu().numpy()
                # print(f"post_sig_action {post_sig_action}")
        ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
        local_size = (int(ground_truth_size[0] / 2) ,int(ground_truth_size[1] / 2))  # (h,w) # TODO 2 is a downsize parameter
        lmb = self.get_local_map_boundaries(self.robot_position, local_size, ground_truth_size)
        target_position = np.array([int(post_sig_action[1] * 320 + lmb[2]), int(post_sig_action[0] * 240 + lmb[0])]) # [x,y]
        # print(f"targ_pos {target_position}")
        return target_position

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(self.max_timestep):
            # print(f"\nstep: {i}")
            self.save_observations(observations)
            value, action_features, action_log_probs = self.act(observations, self.actor)
            self.save_action(action_features, action_log_probs)

            # find target position from action features
            target_position = self.find_target_pos(action_features)
            # find closest node to target position
            target_node_index = self.env.find_index_from_coords(target_position)
            # find coordinates of target nod
            target_node_position = self.env.node_coords[target_node_index]

            # use a star to find shortest path to target node
            dist, route = self.env.graph_generator.find_shortest_path(self.robot_position, target_node_position, self.env.node_coords)
            if route == []: # remain at same pos if destination same as target
                next_position = self.robot_position
            # NOTE can have a better way to do this, ie find closest point?
            elif route == None: # IF unreachable, stay at the same place
                next_position = self.robot_position
            else:   # go to next node in path planned by a star
                next_position = self.env.node_coords[int(route[1])]
            # print(f"next_pos {next_position}")

            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position, target_position, self.travel_dist)
            self.save_reward_done(reward, done)

            observations = self.get_observations()

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            if done:
                break
        
        self.episode_len.append(i+1)

        # print(self.episode_obs)
        # print(self.episode_acts)
        # print(self.episode_log_probs)
        # print(self.episode_rewards)
        # print(self.episode_returns)
        # print(self.episode_len)

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)


    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

# ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
# local_size = (int(ground_truth_size[0] / 2) ,int(ground_truth_size[1] / 2))  # (h,w) # TODO 2 is a downsize parameter

# global_policy = Global_Policy((8,240,320), 256)
# worker = Worker(19, global_policy,save_image=True)
# worker.work(19)