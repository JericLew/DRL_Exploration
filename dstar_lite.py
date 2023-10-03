import heapq
from my_graph import Node, Graph

class DStarLite:
    def __init__(self):
        self.graph = Graph()
        self.queue = []
        self.start_id = None
        self.goal_id = None
        self.k_m = 0

    def initDStarLite(self, graph, start_id, goal_id):
        self.graph = graph
        self.goal_id = goal_id
        self.start_id = start_id
        self.k_m = 0
        self.graph.nodes[goal_id].rhs = 0
        heapq.heappush(self.queue, self.calculateKey(goal_id) + (goal_id,))
        self.computeShortestPath()

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

    def calculateKey(self, index):
        return (min(self.graph.nodes[index].g, self.graph.nodes[index].rhs) + self.h(index, self.start_id) + self.k_m,
                min(self.graph.nodes[index].g, self.graph.nodes[index].rhs))


    def updateVertex(self, index):
        if index != self.goal_id:
            min_rhs = float('inf')
            for (neighbour_id, neighbour_cost) in self.graph.nodes[index].neighbours.items():
                min_rhs = min(min_rhs, self.graph.nodes[neighbour_id].g + neighbour_cost)
            self.graph.nodes[index].rhs = min_rhs
        id_in_queue = [item for item in self.queue if index in item]
        if id_in_queue != []:
            if len(id_in_queue) != 1:
                raise ValueError('more than one ' + index + ' in the queue!')
            self.queue.remove(id_in_queue[0]) # TODO CHECK QUEUE IMPL
        if self.graph.nodes[index].rhs != self.graph.nodes[index].g:
            heapq.heappush(self.queue, self.calculateKey(index) + (index,))


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
                self.graph.nodes[u].g = float('inf')
                self.updateVertex(u)
                for neighbour_id in self.graph.nodes[u].neighbours.keys():
                    self.updateVertex(neighbour_id)
            # graph.printGValues()


    def nextInShortestPath(self):
        min_rhs = float('inf')
        next_start_id = None
        if self.graph.nodes[self.start_id].rhs == float('inf'):
            # print('You are stuck')
            return next_start_id
        else:
            for (neighbour_id, neighbour_cost) in self.graph.nodes[self.start_id].neighbours.items():
                neighbour_cost = self.graph.nodes[neighbour_id].g + neighbour_cost
                if (neighbour_cost) < min_rhs:
                    min_rhs = neighbour_cost
                    next_start_id = neighbour_id
            if next_start_id:
                return next_start_id
            else:
                # raise ValueError('could not find child for transition!')
                # print("Could not find next neighbour")
                return next_start_id     