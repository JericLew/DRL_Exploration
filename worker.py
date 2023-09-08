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
    def __init__(self, global_step, network, device='cuda', save_image=False):
        self.device = device
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)

        self.network = network
        self.dist = DiagGaussian(self.network.output_size, 2) # 256, 2
        # 2 is from box environment defined, cos x and y coord vector space

        # self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = [] # rollout
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

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
        
        global_map = torch.zeros(4, ground_truth_size[0], ground_truth_size[1])
        local_map = torch.zeros(4, local_size[0], local_size[1])
        observations = torch.zeros(8, local_size[0], local_size[1]) # (8,height,width)

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

        observations[0:4, :, :] = local_map
        observations[4:, :, :] = nn.MaxPool2d(2)(global_map)

        # # map check uncomment to check output of observation
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(robot_belief, cmap='gray')
        # axes[1].imshow(global_map[2, : :], cmap='gray') 
        # axes[2].imshow(local_map[2, : :], cmap='gray')
        # plt.savefig('output.png')

        return observations
    
    def act(self, observations, network): # TODO change this
        value, actor_features = network(observations)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs

    # def save_observations(self, observations):
    #     node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
    #     self.episode_buffer[0] += copy.deepcopy(node_inputs)
    #     self.episode_buffer[1] += copy.deepcopy(edge_inputs)
    #     self.episode_buffer[2] += copy.deepcopy(current_index)
    #     self.episode_buffer[3] += copy.deepcopy(node_padding_mask)
    #     self.episode_buffer[4] += copy.deepcopy(edge_padding_mask)
    #     self.episode_buffer[5] += copy.deepcopy(edge_mask)

    # def save_action(self, action_index):
    #     self.episode_buffer[6] += action_index.unsqueeze(0).unsqueeze(0)

    # def save_reward_done(self, reward, done):
    #     self.episode_buffer[7] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
    #     self.episode_buffer[8] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    # def save_next_observations(self, observations):
    #     node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
    #     self.episode_buffer[9] += copy.deepcopy(node_inputs)
    #     self.episode_buffer[10] += copy.deepcopy(edge_inputs)
    #     self.episode_buffer[11] += copy.deepcopy(current_index)
    #     self.episode_buffer[12] += copy.deepcopy(node_padding_mask)
    #     self.episode_buffer[13] += copy.deepcopy(edge_padding_mask)
    #     self.episode_buffer[14] += copy.deepcopy(edge_mask)

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            print(f"\nstep: {i}")
            value, raw_action, action_log_probs = self.act(observations, self.network)

            post_sig_action = nn.Sigmoid()(raw_action).cpu().numpy()
            print(f"post_sig_action {post_sig_action}")
            ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
            local_size = (int(ground_truth_size[0] / 2) ,int(ground_truth_size[1] / 2))  # (h,w) # TODO 2 is a downsize parameter
            lmb = self.get_local_map_boundaries(self.robot_position, local_size, ground_truth_size)
            target_position = np.array([int(post_sig_action[0][1] * 320 + lmb[2]), int(post_sig_action[0][0] * 240 + lmb[0])]) # [x,y]
            print(f"targ_pos {target_position}")


            # if len(self.env.frontiers) != 0:
            best_frontier = self.env.frontiers[0]
            min_distance = np.linalg.norm(target_position - best_frontier)

            for frontier_idx in range(1, len(self.env.frontiers)):
                distance_to_current = np.linalg.norm(target_position - self.env.frontiers[frontier_idx])
                if distance_to_current < min_distance:
                    best_frontier = self.env.frontiers[frontier_idx]
                    min_distance = distance_to_current

            next_position = best_frontier
            #     next_position = self.robot_position

            print(f"next_pos {next_position}")

            self.get_observations()
            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position, target_position, self.travel_dist)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            if done:
                break

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
        # for filename in self.env.frame_files[:-1]:
        #     os.remove(filename)

# ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
# local_size = (int(ground_truth_size[0] / 2) ,int(ground_truth_size[1] / 2))  # (h,w) # TODO 2 is a downsize parameter

global_policy = Global_Policy((8,240,320), 256)
worker = Worker(1, global_policy,save_image=True)
worker.work(1)