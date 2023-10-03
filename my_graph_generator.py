import numpy as np
import copy

from parameter import *
from dstar_lite import*
from my_graph import Node, Graph

# TODO optimise graph generation and update

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
        self.node_ids = []
        self.dstar_driver = DStarLite()

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        y = np.linspace(0, self.map_y - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points
    
    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords
    
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
            self.node_ids.append(node.index)
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

    def update_graph(self, robot_belief, old_robot_belief):
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
            self.dstar_driver.updateVertex(node.index)
            
            if self.plot:
                for neighbour_id in node.neighbours.keys():
                    self.x.append([node.coord[0], self.graph.nodes[neighbour_id].coord[0]])
                    self.y.append([node.coord[1], self.graph.nodes[neighbour_id].coord[1]])
        # return self.graph

    def reset_dstar_values(self):
        for node in self.graph.nodes.values():
            node.g = float("inf")
            node.rhs = float("inf")
        # return self.graph