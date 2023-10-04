import copy
import numpy as np

from parameter import *
from dstar_lite import*
from my_graph import Node, Graph

# TODO optimise graph generation and update for speed using numpy

class Graph_generator:
    def __init__(self, map_size, plot=False):
        self.graph = Graph()
        self.plot = plot
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.x = []
        self.y = []
        self.node_coords = None
        self.node_ids = np.array([], dtype=str)
        self.dstar_driver = DStarLite()

    def find_node_id_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        node_id = self.node_ids[index]
        return node_id
    
    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        y = np.linspace(0, self.map_y - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points
    
    def reset_dstar_values(self):
        for node in self.graph.nodes.values():
            node.g = float("inf")
            node.rhs = float("inf")
        # return self.graph
       
    def area_in_range(self, robot_position, sensor_range):
        x0, y0 = robot_position
        x_max, y_max = self.map_x, self.map_y  # Replace with actual dimensions
        x_range, y_range = np.arange(x_max), np.arange(y_max)
        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
        
        # Calculate the distance from the center to each point on the map
        distances = np.sqrt((xx - x0)**2 + (yy - y0)**2)
        
        # Create a mask for points within the radius
        mask = distances <= sensor_range
        
        # Get the coordinates within the radius
        coordinates = np.column_stack((xx[mask], yy[mask]))
        
        return coordinates
    
    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            # if k == 127: # REMOVED COLLISION IN UNEXPLORED
            #     collision = True
            #     break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision
    
    def generate_graph(self, robot_belief): 
        node_coords = self.uniform_points[:]
        self.node_coords = node_coords
        # GENERATE NODES
        for i , coord in enumerate(node_coords):
            x = i // UNIFORM_POINT_INTERVAL
            y = i % UNIFORM_POINT_INTERVAL
            node = Node(f"x{x}y{y}", x, y, coord)
            self.node_ids = np.append(self.node_ids, node.index)
            self.graph.nodes[node.index] = node
        # GENERATE NEIGHBOURS (8-Connected)
        for node in self.graph.nodes.values():
            x, y = node.x, node.y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < UNIFORM_POINT_INTERVAL and 0 <= ny < UNIFORM_POINT_INTERVAL:
                        neighbour_id = f"x{nx}y{ny}"
                        neighbour_coord = self.graph.nodes[neighbour_id].coord
                        if self.check_collision(node.coord, neighbour_coord, robot_belief):
                            cost = float("inf")
                        else:
                            cost = np.linalg.norm(node.coord - neighbour_coord)
                        node.neighbours[neighbour_id] = cost

            if self.plot:
                for neighbour_id in node.neighbours.keys():
                    self.x.append([node.coord[0], self.graph.nodes[neighbour_id].coord[0]])
                    self.y.append([node.coord[1], self.graph.nodes[neighbour_id].coord[1]])
        # return self.graph

    def update_graph(self, robot_belief, robot_position, sensor_range):
        area_in_range = self.area_in_range(robot_position, sensor_range)
        area_to_check = area_in_range[:, 0] + area_in_range[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(area_to_check, uniform_points_to_check, return_indices=True)
        node_coords_to_check = self.uniform_points[candidate_indices]
        indices = []
        for value in node_coords_to_check:
            matching_indices = np.where((self.node_coords == value).all(axis=1))[0]
            indices.extend(matching_indices)
        node_ids_to_check = self.node_ids[indices]
        # print(f"current pos {self.find_node_id_from_coords(robot_position)}")
        # print(f"node_ids_to_check {node_ids_to_check}")

        for node_ids in node_ids_to_check:
            node = self.graph.nodes[node_ids]
            x, y = node.x, node.y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < UNIFORM_POINT_INTERVAL and 0 <= ny < UNIFORM_POINT_INTERVAL:
                        neighbour_id = f"x{nx}y{ny}"
                        neighbour_coord = self.graph.nodes[neighbour_id].coord
                        if self.check_collision(node.coord, neighbour_coord, robot_belief):
                            cost = float("inf")
                        else:
                            cost = np.linalg.norm(node.coord - neighbour_coord)
                        node.neighbours[neighbour_id] = cost
            self.dstar_driver.updateVertex(node.index)
        # return self.graph