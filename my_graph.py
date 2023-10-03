class Node:
    def __init__(self, id, x, y, coord):
        self.id = id
        self.x = x
        self.y = y
        self.coord = coord
        # dictionary of neighbour node ID's
        # key = id of neighbour
        # value = (edge cost,)
        self.neighbours = dict()

        # g approximation
        self.g = float('inf')
        # rhs value
        self.rhs = float('inf')

    def __str__(self):
        return 'Node: ' + self.id + ' g: ' + str(self.g) + ' rhs: ' + str(self.rhs)

    def __repr__(self):
        return self.__str__()

class Graph:
    def __init__(self):
        # key is ID, value is Node object
        self.nodes = dict()

    def __str__(self):
        msg = 'Graph:'
        for (node_id, node_object) in self.nodes.items:
            msg += '\n  node: ' + node_id + ' g: ' + \
                str(node_object.g) + ' rhs: ' + str(node_object.rhs)
        return msg

    def __repr__(self):
        return self.__str__()

    # def setStart(self, id):
    #     if(self.graph[id]):
    #         self.start = id
    #     else:
    #         raise ValueError('start id not in graph')

    # def setGoal(self, id):
    #     if(self.graph[id]):
    #         self.goal = id
    #     else:
    #         raise ValueError('goal id not in graph')