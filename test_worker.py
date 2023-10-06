import os
import csv
import copy
import torch
import imageio
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn

from env import Env
from test_parameter import *

class TestWorker:
    def __init__(self, meta_agent_id, actor_critic, global_step, save_image=False):
        # Handle devices for global training and local simulation
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')

        # Initialise local actor critic for simulation
        self.actor_critic = actor_critic

        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.max_timestep = MAX_TIMESTEP_PER_EPISODE
        self.save_image = save_image
        self.env = Env(map_index=self.global_step, plot=save_image, test=True)
       
        # Initialise varibles
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.perf_metrics = dict()

    # Function to get corner coords for robot local area
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
    
    # Retrieve observation with shape (8 x Local H x Local W)
    def get_observations(self):
        # observation[0, :, :] probability of obstacle
        # observation[1, :, :] probability of exploration
        # observation[2, :, :] indicator of current position
        # observation[3, :, :] indicator of visited

        # TODO make it less computationally intensive
        robot_belief = copy.deepcopy(self.env.robot_belief)
        visited_map = copy.deepcopy(self.env.visited_map)
        ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
        local_size = (int(ground_truth_size[0] / MAP_DOWNSIZE_FACTOR), \
                      int(ground_truth_size[1] / MAP_DOWNSIZE_FACTOR)) # (h,w)
        
        global_map = torch.zeros(4, ground_truth_size[0], ground_truth_size[1]).to(self.device)
        local_map = torch.zeros(4, local_size[0], local_size[1]).to(self.device)
        observations = torch.zeros(8, local_size[0], local_size[1]).to(self.device) # (8,height,width)

        lmb = self.get_local_map_boundaries(self.robot_position, local_size, ground_truth_size)

        # Create a mask for each condition
        mask_obst = (robot_belief == 1) # if colour 1 : index 0 = 1, index 1 = 1 obst
        mask_free = (robot_belief == 255) # if colour 255: index 0 = 0, index 1 = 1 free
        mask_unkn = (robot_belief == 127) # if colour 127: index 0 = 0, index 1 = 0 unkw
        mask_visi = (visited_map == 1) # if visited: index : 3 = 1 vist

        # Update global map based on the masks
        global_map[0, mask_obst] = 1
        global_map[1, mask_obst] = 1
        global_map[0, mask_free] = 0
        global_map[1, mask_free] = 1
        global_map[0, mask_unkn] = 0
        global_map[1, mask_unkn] = 0
        global_map[3, mask_visi] = 1
        global_map[2, self.robot_position[1] - 4:self.robot_position[1] + 5, \
                    self.robot_position[0] - 4:self.robot_position[0] + 5] = 1 
        
        local_map = global_map[:, lmb[0]:lmb[1], lmb[2]:lmb[3]] # (width,height)

        observations[0:4, :, :] = local_map.detach()
        observations[4:, :, :] = nn.MaxPool2d(MAP_DOWNSIZE_FACTOR)(global_map)

        '''map check uncomment to check output of observation'''
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(robot_belief, cmap='gray')
        # axes[1].imshow(global_map[3, : :], cmap='gray') 
        # axes[2].imshow(local_map[3, : :], cmap='gray')
        # plt.savefig('output.png')
        return observations
    
    # Process actor output to target position
    def find_target_pos(self, action):
        with torch.no_grad():
            post_sig_action = nn.Sigmoid()(action).cpu().numpy()
        ground_truth_size = copy.deepcopy(self.env.ground_truth_size)  # (480, 640)
        local_size = (int(ground_truth_size[0] / MAP_DOWNSIZE_FACTOR),\
                      int(ground_truth_size[1] / MAP_DOWNSIZE_FACTOR))  # (h,w)
        lmb = self.get_local_map_boundaries(self.robot_position, local_size, ground_truth_size)
        target_position = np.array([int(post_sig_action[1] * 320 + lmb[2]), int(post_sig_action[0] * 240 + lmb[0])]) # [x,y]
        return target_position
    
    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        self.save_observations(observations)
        value, action, action_log_probs = self.actor_critic.act(observations)
        target_position = self.find_target_pos(action)

        start_id = self.env.find_node_id_from_coords(self.robot_position)
        goal_id = self.env.find_node_id_from_coords(target_position)
        self.env.graph_generator.dstar_driver.initDStarLite(self.env.graph_generator.graph, start_id, goal_id)

        reward = 0

        for num_step in range(self.max_timestep):
            planning_step = num_step // NUM_ACTION_STEP
            action_step = num_step % NUM_ACTION_STEP

            self.env.graph_generator.dstar_driver.computeShortestPath()
            next_position_id = self.env.graph_generator.dstar_driver.nextInShortestPath()

            if next_position_id: # CHECK IF NEXT COORD CAN BE FOUND
                next_position = self.env.graph_generator.graph.nodes[next_position_id].coord
            else:
                next_position = self.robot_position

            step_reward, done, self.robot_position, self.travel_dist =\
                self.env.step(self.robot_position, next_position, target_position, self.travel_dist)
            reward += step_reward


            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, num_step, self.travel_dist)

            # save evaluation data
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = f'{trajectory_path}/ours_trajectory_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist', 'area']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist, np.sum(self.env.robot_belief == 255)]).reshape(1, -1)
                    writer.writerows(csv_data)

            # At last action step do global selection
            if action_step == NUM_ACTION_STEP - 1 or done:
                reward = 0

                if done or planning_step == NUM_PLANNING_STEP - 1:
                    break

                observations = self.get_observations()
                self.save_observations(observations)
                value, action, action_log_probs = self.actor_critic.act(observations)
                target_position = self.find_target_pos(action)

                start_id = self.env.find_node_id_from_coords(self.robot_position)
                goal_id = self.env.find_node_id_from_coords(target_position)
                self.env.graph_generator.reset_dstar_values()
                self.env.graph_generator.dstar_driver.initDStarLite(self.env.graph_generator.graph, start_id, goal_id)

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save final path length
        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = f'{length_path}/ours_length_result.csv'
            new_file = False if os.path.exists(csv_filename) else True
            field_names = ['dist']
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.travel_dist]).reshape(-1,1)
                writer.writerows(csv_data)

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, curr_episode):
        self.run_episode(curr_episode)

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

