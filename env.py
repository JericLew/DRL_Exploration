import os
import copy
import matplotlib.pyplot as plt

from skimage import io
from skimage.measure import block_reduce

from sensor import *
from parameter import *
from my_graph_generator import *

class Env():
    def __init__(self, map_index, plot=False, test=False):
        # import environment ground truth from dungeon files
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/complex'  # change to 'complex', 'medium', and 'easy'
        else:
            self.map_dir = f'DungeonMaps/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position = self.import_ground_truth(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth)  # (480, 640)

        # initialize parameters
        self.resolution = 4 # to downsample the map for frontier
        self.sensor_range = 80
        self.explored_rate = 0

        # initialize robot_belief
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # Unexplored = 127
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        # initialize graph generator
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, plot=plot)

        # Arrays for global input update
        self.frontiers = None
        self.visited_map = np.zeros(self.ground_truth_size)
        self.visited_map[self.start_position[1] - 4:self.start_position[1] + 5,\
                        self.start_position[0] - 4:self.start_position[0] + 5] = 1
        self.visited = np.array([self.start_position])
        self.targets = np.array([self.start_position])

        self.begin()

        # plot related
        self.plot = plot
        self.frame_files = []

    def find_node_id_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.graph_generator.node_coords - position, axis=1))
        node_id = self.graph_generator.node_ids[index]
        return node_id

    def begin(self):
        self.robot_belief = self.update_robot_belief(self.start_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)\

        # downsampled belief has lower resolution than robot belief
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.graph_generator.generate_graph(self.robot_belief)

    def step(self, robot_position, next_position, target_position, travel_dist):
        
        # check if havent reach goal
        robot_position_id = self.find_node_id_from_coords(robot_position)
        target_position_id = self.find_node_id_from_coords(target_position)
        if robot_position_id != target_position_id:
            # Increment k_m and change start_id
            self.graph_generator.dstar_driver.k_m +=\
                self.graph_generator.dstar_driver.h(
                    self.find_node_id_from_coords(robot_position),
                    self.find_node_id_from_coords(next_position))
            self.graph_generator.dstar_driver.start_id = self.find_node_id_from_coords(next_position)
        # if reach goal, dont move until target changes
        else:
            next_position = robot_position

        # Calculate and update total dist
        dist = np.linalg.norm(robot_position - next_position)
        travel_dist += dist
        
        # move the robot to the selected position and update its belief
        robot_position = next_position
        self.robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)

        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()

        # calculate the reward associated with the action
        reward = self.calculate_reward(dist, frontiers)

        self.visited_map[robot_position[1] - 4:robot_position[1] + 5,\
                        robot_position[0] - 4:robot_position[0] + 5] = 1 # for masking to update observation
    
        self.visited = np.append(self.visited, [robot_position], axis=0) # can be faster?
        self.targets = np.append(self.targets, [target_position], axis = 0)

        # update graph
        self.graph_generator.update_graph(self.robot_belief, robot_position, self.sensor_range)

        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.frontiers = frontiers

        # check if done
        done = self.check_done()
        if done:
            reward += FINISHING_REWARD * REWARD_SCALE_FACTOR

        return reward, done, robot_position, travel_dist

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def check_done(self):
        done = False
        # if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
        if self.test and np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            done = True
        elif len(self.frontiers) == 0:
            done = True
        return done
    
    def calculate_reward(self, dist, frontiers):
        reward = 0

        # check the num of observed frontiers
        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = self.frontiers[:, 0] + self.frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        reward -= dist / DIST_DENOMINATOR
        reward += delta_num / FRONTIER_DENOMINATOR
        if dist == 0:
            reward -= SAME_POSITION_PUNISHMENT

        # print(f"dist {dist}, delta num {delta_num}, reward {reward}, scaled reward {reward * REWARD_SCALE_FACTOR}")
        return reward * REWARD_SCALE_FACTOR

    def evaluate_exploration_rate(self):
        rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return rate
    
    def calculate_new_free_area(self):
        old_free_area = self.old_robot_belief == 255
        current_free_area = self.robot_belief == 255
        new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int))

        return np.sum(new_free_area)

    def find_frontier(self):
        # find frontiers from downsampled_belief by checking nearby 8 cells for each cell
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution

        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        # plt.ion()
        plt.cla()
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #     plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        # plt.scatter(self.graph_generator.node_coords[:, 0], self.graph_generator.node_coords[:, 1], c='b', zorder=5)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.plot(self.targets[-5:, 0], self.targets[-5:, 1], 'g--', linewidth=2)
        plt.plot(self.targets[-1, 0], self.targets[-1, 1], 'gx', markersize=8)
        plt.plot(self.visited[:, 0], self.visited[:, 1], 'b', linewidth=2)
        plt.plot(self.visited[-1, 0], self.visited[-1, 1], 'mo', markersize=8)
        plt.plot(self.visited[0, 0], self.visited[0, 1], 'co', markersize=8)
        # plt.pause(0.1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, travel_dist))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
