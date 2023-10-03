import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

from parameter import *
from node import Node
from graph import Graph, a_star
import heapq

class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []

        self.queue = []
        self.start_id = None
        self.goal_id = None
        self.k_m = 0


    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []

    # def generate_graph(self, robot_location, robot_belief):
    #     # get node_coords by finding the uniform points in free area
    #     free_area = self.free_area(robot_belief)
    #     free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
    #     uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
    #     _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
    #     node_coords = self.uniform_points[candidate_indices]
    #     # add robot location as one node coords
    #     node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
    #     self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)

    #     # generate the collision free graph
    #     self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

    #     return self.node_coords, self.graph.edges

    def generate_graph(self, robot_location, robot_belief):
        node_coords = self.uniform_points[:]
        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)
        # generate the neighbours for graphs
        self.generate_edges(self.node_coords, robot_belief)

        return self.node_coords, self.graph.edges

    def generate_edges(self, node_coords, robot_belief):
        X = node_coords
        # for id, node_coord in enumerate(X):
        #     row = id % 30
        #     col = id // 30

        #     a = str(self.find_index_from_coords(node_coords, node_coord))
        #     self.graph.add_node(a)
        #     self.graph.g_values[int(a)] = float("inf")
        #     self.graph.rhs_values[int(a)] = float("inf")
        #     if row > 0:
        #         neighbour_coord = node_coords[id - 1]
        #         b = str(self.find_index_from_coords(node_coords, neighbour_coord))
        #     if row + 1 < 30:
        #     if col > 0:
        #     if col + 1 < 30:

        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                if j == 0:
                    continue
                start = p
                end = neighbour
                a = str(self.find_index_from_coords(node_coords, p))
                b = str(self.find_index_from_coords(node_coords, neighbour))
                self.graph.add_node(a)
                self.graph.g_values[int(a)] = float("inf")
                self.graph.rhs_values[int(a)] = float("inf")
                if self.check_collision(start, end, robot_belief):
                    self.graph.add_edge(a, b, float("inf"))
                    self.graph.add_edge(b, a, float("inf"))
                else:
                    self.graph.add_edge(a, b, distances[i, j])
                    self.graph.add_edge(b, a, distances[i, j])


                if self.plot:
                    self.x.append([p[0], neighbour[0]])
                    self.y.append([p[1], neighbour[1]])

    def update_edges(self, new_free_node_coords, robot_belief):
        print("updating edges")
        X = new_free_node_coords
        for node_coord in X:
            node_id = str(self.find_index_from_coords(self.node_coords, node_coord))
            for edge in self.graph.edges[node_id].values():
                neighbour_id = int(edge.to_node)
                neighbour_coord = self.node_coords[neighbour_id]
                if self.check_collision(node_coord, neighbour_coord, robot_belief):
                    edge.length = float("inf")
                    self.graph.edges[str(neighbour_id)][node_id].length = float("inf")
            self.updateVertex(int(node_id))
            

    def update_graph(self, robot_belief, old_robot_belief):
        # add uniform points in the new free area to the node coords
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_free_node_coords = self.uniform_points[candidate_indices]
        self.update_edges(new_free_node_coords, robot_belief) # TODO WORKS IF I CHANGE UPDATE TO ALL 

        return self.node_coords, self.graph.edges
    
    # def update_graph(self, robot_belief, old_robot_belief):
    #     # add uniform points in the new free area to the node coords
    #     new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
    #     free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
    #     uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
    #     _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
    #     new_node_coords = self.uniform_points[candidate_indices]
    #     self.node_coords = np.concatenate((self.node_coords, new_node_coords))

    #     self.edge_clear_all_nodes()
    #     self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)

    #     return self.node_coords, self.graph.edges

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        y = np.linspace(0, self.map_y - 1, UNIFORM_POINT_INTERVAL).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free
    
    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])

                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

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

    def find_shortest_path(self, current, destination, node_coords):
        start_node = str(self.find_index_from_coords(node_coords, current))
        end_node = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(start_node), int(end_node), self.node_coords, self.graph)
        if start_node != end_node:
            assert route != []
        if route == None:
            return dist, route
        route = list(map(str, route))
        return dist, route

    def topKey(self):
        self.queue.sort()
        # print(queue)
        if len(self.queue) > 0:
            return self.queue[0][:2]
        else:
            # print('empty queue!')
            return (float('inf'), float('inf'))

    def h(self, id, destination):
        current = self.node_coords[id]
        end = self.node_coords[destination]
        # h = abs(end[0] - current[0]) + abs(end[1] - current[1])
        h = max(abs(end[0] - current[0]), abs(end[1] - current[1]))
        # h = ((end[0]-current[0])**2 + (end[1] - current[1])**2)**(1/2)
        return h

    def calculateKey(self, id):
        return (min(self.graph.g_values[id], self.graph.rhs_values[id]) + self.h(id, self.start_id) + self.k_m,
                min(self.graph.g_values[id], self.graph.rhs_values[id]))


    def updateVertex(self, id):
        if id != self.goal_id:
            min_rhs = float('inf')
            for edge in self.graph.edges[str(id)].values():
                neighbour_id = int(edge.to_node)
                min_rhs = min(min_rhs, self.graph.g_values[neighbour_id] + edge.length)
            # for i in graph.graph[id].children:
            #     min_rhs = min( # TODO find min RHS of successor
            #         min_rhs, graph.graph[i].g + graph.graph[id].children[i])
            self.graph.rhs_values[id] = min_rhs
        id_in_queue = [item for item in self.queue if id in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + str(id) + ' in the queue!')
            self.queue.remove(id_in_queue[0]) # TODO CHECK QUEUE IMPL
        if self.graph.rhs_values[id] != self.graph.g_values[id]:
            heapq.heappush(self.queue, self.calculateKey(id) + (id,))


    def computeShortestPath(self):
        while (self.graph.rhs_values[self.start_id] != self.graph.g_values[self.start_id]) or (self.topKey() < self.calculateKey(self.start_id)):
            # print(graph.graph[s_start])
            # print('topKey')
            # print(topKey(queue))
            # print('calculateKey')
            # print(calculateKey(graph, s_start, 0))
            k_old = self.topKey()
            u = heapq.heappop(self.queue)[2] # current ID
            if k_old < self.calculateKey(u):
                heapq.heappush(self.queue, self.calculateKey(u) + (u,))
            elif self.graph.g_values[u] > self.graph.rhs_values[u]:
                self.graph.g_values[u] = self.graph.rhs_values[u]
                for edge in self.graph.edges[str(u)].values():
                    neighbour_id = int(edge.to_node)
                    self.updateVertex(neighbour_id)
                # for id in graph.graph[u].parents: #TODO PREDescsor
                #     updateVertex(graph, queue, id, start_id, goal_id, node_coords, k_m)
            else:
                self.graph.g_values[u] = float('inf')
                self.updateVertex(u)
                for edge in self.graph.edges[str(u)].values():
                    neighbour_id = int(edge.to_node)
                    self.updateVertex(neighbour_id)
                # for id in self.graph.graph[u].parents: #TODO PREDescsor
                #     updateVertex(id)
            # graph.printGValues()


    def nextInShortestPath(self):
        min_rhs = float('inf')
        s_next = None
        if self.graph.rhs_values[self.start_id] == float('inf'):
            print('You are done stuck')
        else:
            print(self.start_id)
            for edge in self.graph.edges[str(self.start_id)].values():
                neighbour_id = int(edge.to_node)
                print(neighbour_id)
                neighbour_cost = self.graph.g_values[neighbour_id] + edge.length
                if (neighbour_cost) < min_rhs:
                    min_rhs = neighbour_cost
                    next_start_id = neighbour_id
            if next_start_id:
                return next_start_id
            # for i in graph.graph[s_current].children:
            #     # print(i)
            #     child_cost = graph.graph[i].g + graph.graph[s_current].children[i]
            #     # print(child_cost)
            #     if (child_cost) < min_rhs:
            #         min_rhs = child_cost
            #         s_next = i
            # if s_next:
            #     return s_next
            else:
                raise ValueError('could not find child for transition!')
            
    def initDStarLite(self, start_id, goal_id):
        self.goal_id = goal_id
        self.start_id = start_id
        self.k_m = 0
        self.graph.rhs_values[goal_id] = 0
        heapq.heappush(self.queue, self.calculateKey(goal_id) + (goal_id,))
        self.computeShortestPath()