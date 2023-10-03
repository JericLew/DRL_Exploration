import numpy as np
import copy

from parameter import *
from my_graph import Node, Graph
import heapq

class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        self.graph = Graph()
        self.plot = plot
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.x = []
        self.y = []
        self.node_coords = None
        self.node_ids = []

        self.route_node = []

        self.queue = []
        self.start_id = None
        self.goal_id = None
        self.k_m = 0

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
    
    def generate_graph(self, robot_location, robot_belief):
        node_coords = self.uniform_points[:]
        node_coords = self.unique_coords(node_coords).reshape(-1, 2) #TODO IDK IF NEED
        self.node_coords = node_coords
        # GENERATE NODES
        for i , coord in enumerate(node_coords):
            x = i // UNIFORM_POINT_INTERVAL
            y = i % UNIFORM_POINT_INTERVAL
            node = Node(f"x{x}y{y}", x, y, coord)
            self.node_ids.append(node.id)
            self.graph.nodes[node.id] = node
        # GENERATE NEIGHBOURS
        for node in self.graph.nodes.values():
            if node.x > 0:
                neighbour_id = f"x{node.x - 1}y{node.y}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.x + 1 < UNIFORM_POINT_INTERVAL:
                neighbour_id = f"x{node.x + 1}y{node.y}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.y > 0:
                neighbour_id = f"x{node.x}y{node.y - 1}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.y + 1 < UNIFORM_POINT_INTERVAL:
                neighbour_id = f"x{node.x}y{node.y + 1}"
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
        return self.graph

    def update_graph(self, robot_belief, old_robot_belief):
        for node in self.graph.nodes.values():
            if node.x > 0:
                neighbour_id = f"x{node.x - 1}y{node.y}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.x + 1 < UNIFORM_POINT_INTERVAL:
                neighbour_id = f"x{node.x + 1}y{node.y}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.y > 0:
                neighbour_id = f"x{node.x}y{node.y - 1}"
                neighbour_coord = self.graph.nodes[neighbour_id].coord
                if self.check_collision(node.coord, neighbour_coord, robot_belief):
                    cost = float("inf")
                else:
                    cost = np.linalg.norm(node.coord - neighbour_coord)
                node.neighbours[neighbour_id] = cost
            if node.y + 1 < UNIFORM_POINT_INTERVAL:
                neighbour_id = f"x{node.x}y{node.y + 1}"
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
        return self.graph
    
    def topKey(self):
        self.queue.sort()
        # print(queue)
        if len(self.queue) > 0:
            return self.queue[0][:2]
        else:
            # print('empty queue!')
            return (float('inf'), float('inf'))

    def h(self, cuurent_id, end_id):
        current_coord = self.graph.nodes[cuurent_id].coord
        end_coord = self.graph.nodes[end_id].coord
        # h = abs(end_coord[0] - current_coord[0]) + abs(end_coord[1] - current_coord[1])
        # h = ((end_coord[0]-current_coord[0])**2 + (end_coord[1] - current_coord[1])**2)**(1/2)
        h = max(abs(end_coord[0] - current_coord[0]), abs(end_coord[1] - current_coord[1]))
        return h

    def calculateKey(self, id):
        return (min(self.graph.nodes[id].g, self.graph.nodes[id].rhs) + self.h(id, self.start_id) + self.k_m,
                min(self.graph.nodes[id].g, self.graph.nodes[id].rhs))


    def updateVertex(self, id):
        if id != self.goal_id:
            min_rhs = float('inf')
            for (neighbour_id, neighbour_cost) in self.graph.nodes[id].neighbours.items():
                min_rhs = min(min_rhs, self.graph.nodes[neighbour_id].g + neighbour_cost)
            self.graph.nodes[id].rhs = min_rhs
        id_in_queue = [item for item in self.queue if id in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + id + ' in the queue!')
            self.queue.remove(id_in_queue[0]) # TODO CHECK QUEUE IMPL
        if self.graph.nodes[id].rhs != self.graph.nodes[id].g:
            heapq.heappush(self.queue, self.calculateKey(id) + (id,))


    def computeShortestPath(self):
        while (self.graph.nodes[self.start_id].rhs != self.graph.nodes[self.start_id].g) or (self.topKey() < self.calculateKey(self.start_id)):
            # print(graph.graph[s_start])
            # print('topKey')
            # print(topKey(queue))
            # print('calculateKey')
            # print(calculateKey(graph, s_start, 0))
            k_old = self.topKey()
            u = heapq.heappop(self.queue)[2] # current ID
            if k_old < self.calculateKey(u):
                heapq.heappush(self.queue, self.calculateKey(u) + (u,))
            elif self.graph.nodes[u].g > self.graph.nodes[u].rhs:
                self.graph.nodes[u].g = self.graph.nodes[u].rhs
                for neighbour_id in self.graph.nodes[u].neighbours.keys():
                    self.updateVertex(neighbour_id)
            else:
                self.graph.g_values[u] = float('inf')
                self.updateVertex(u)
                for neighbour_id in self.graph.nodes[id].neighbours.keys():
                    self.updateVertex(neighbour_id)
            # graph.printGValues()


    def nextInShortestPath(self):
        min_rhs = float('inf')
        next_start_id = None
        if self.graph.nodes[self.start_id].rhs == float('inf'):
            print('You are done stuck')
        else:
            for (neighbour_id, neighbour_cost) in self.graph.nodes[self.start_id].neighbours.items():
                neighbour_cost = self.graph.nodes[neighbour_id].g + neighbour_cost
                if (neighbour_cost) < min_rhs:
                    min_rhs = neighbour_cost
                    next_start_id = neighbour_id
            if next_start_id:
                return next_start_id
            else:
                raise ValueError('could not find child for transition!')
            
    def initDStarLite(self, start_id, goal_id):
        self.goal_id = goal_id
        self.start_id = start_id
        self.k_m = 0
        self.graph.nodes[goal_id].rhs = 0
        heapq.heappush(self.queue, self.calculateKey(goal_id) + (goal_id,))
        self.computeShortestPath()