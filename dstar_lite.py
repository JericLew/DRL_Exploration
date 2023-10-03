import heapq
# from utils import stateNameToCoords


def topKey(queue):
    queue.sort()
    # print(queue)
    if len(queue) > 0:
        return queue[0][:2]
    else:
        # print('empty queue!')
        return (float('inf'), float('inf'))

def h(id, destination, node_coords):
    current = node_coords[id]
    end = node_coords[destination]
    h = abs(end[0] - current[0]) + abs(end[1] - current[1])
    # h = ((end[0]-current[0])**2 + (end[1] - current[1])**2)**(1/2)
    return h

def calculateKey(graph, id, start_id, node_coords, k_m):
    return (min(graph.g_values[id], graph.rhs_values[id]) + h(id, start_id, node_coords) + k_m,
            min(graph.g_values[id], graph.rhs_values[id]))


def updateVertex(graph, queue, id, start_id, goal_id, node_coords, k_m):
    if id != goal_id:
        min_rhs = float('inf')
        for edge in graph.edges[id].values():
            neighbour_id = int(edge.to_node)
            min_rhs = min(min_rhs, graph.g_values[neighbour_id] + edge.length)
        # for i in graph.graph[id].children:
        #     min_rhs = min( # TODO find min RHS of successor
        #         min_rhs, graph.graph[i].g + graph.graph[id].children[i])
        graph.rhs_values[id].rhs = min_rhs
    id_in_queue = [item for item in queue if id in item]
    if id_in_queue != []:
        if len(id_in_queue) != 1:
            raise ValueError('more than one ' + id + ' in the queue!')
        queue.remove(id_in_queue[0]) # TODO CHECK QUEUE IMPL
    if graph.rhs_values[id] != graph.g_values[id]:
        heapq.heappush(queue, calculateKey(graph, id, start_id, node_coords, k_m) + (id,))


def computeShortestPath(graph, queue, start_id, goal_id, node_coords, k_m):
    while (graph.rhs_values[start_id] != graph.g_values[start_id]) or (topKey(queue) < calculateKey(graph, start_id, start_id, node_coords, k_m)):
        # print(graph.graph[s_start])
        # print('topKey')
        # print(topKey(queue))
        # print('calculateKey')
        # print(calculateKey(graph, s_start, 0))
        k_old = topKey(queue)
        u = heapq.heappop(queue)[2] # current ID
        if k_old < calculateKey(graph, u, start_id, node_coords, k_m):
            heapq.heappush(queue, calculateKey(graph, u, start_id, node_coords, k_m) + (u,))
        elif graph.g_values[u] > graph.rhs_values[u]:
            graph.g_values[u] = graph.rhs_values[u]
            for edge in graph.edges[u].values():
                neighbour_id = int(edge.to_node)
                updateVertex(graph, queue, neighbour_id, start_id, goal_id, node_coords, k_m)
            # for id in graph.graph[u].parents: #TODO PREDescsor
            #     updateVertex(graph, queue, id, start_id, goal_id, node_coords, k_m)
        else:
            graph.g_values[u] = float('inf')
            updateVertex(graph, queue, u, start_id, k_m)
            for edge in graph.edges[u].values():
                neighbour_id = int(edge.to_node)
                updateVertex(graph, queue, neighbour_id, start_id, goal_id, node_coords, k_m)
            for id in graph.graph[u].parents: #TODO PREDescsor
                updateVertex(graph, queue, id, start_id, goal_id, node_coords, k_m)
        # graph.printGValues()


def nextInShortestPath(graph, start_id):
    min_rhs = float('inf')
    s_next = None
    if graph.rhs_values[start_id] == float('inf'):
        print('You are done stuck')
    else:
        for edge in graph.edges[start_id].values():
            neighbour_id = int(edge.to_node)
            neighbour_cost = graph.g_values[neighbour_id]
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


def scanForObstacles(graph, queue, s_current, scan_range, k_m):
    states_to_update = {}
    range_checked = 0
    if scan_range >= 1:
        for neighbor in graph.graph[s_current].children:
            neighbor_coords = stateNameToCoords(neighbor)
            states_to_update[neighbor] = graph.cells[neighbor_coords[1]
                                                     ][neighbor_coords[0]]
        range_checked = 1
    # print(states_to_update)

    while range_checked < scan_range:
        new_set = {}
        for state in states_to_update:
            new_set[state] = states_to_update[state]
            for neighbor in graph.graph[state].children:
                if neighbor not in new_set:
                    neighbor_coords = stateNameToCoords(neighbor)
                    new_set[neighbor] = graph.cells[neighbor_coords[1]
                                                    ][neighbor_coords[0]]
        range_checked += 1
        states_to_update = new_set

    new_obstacle = False
    for state in states_to_update:
        if states_to_update[state] < 0:  # found cell with obstacle
            # print('found obstacle in ', state)
            for neighbor in graph.graph[state].children:
                # first time to observe this obstacle where one wasn't before
                if(graph.graph[state].children[neighbor] != float('inf')):
                    neighbor_coords = stateNameToCoords(state)
                    graph.cells[neighbor_coords[1]][neighbor_coords[0]] = -2
                    graph.graph[neighbor].children[state] = float('inf')
                    graph.graph[state].children[neighbor] = float('inf')
                    updateVertex(graph, queue, state, s_current, k_m)
                    new_obstacle = True
        # elif states_to_update[state] == 0: #cell without obstacle
            # for neighbor in graph.graph[state].children:
                # if(graph.graph[state].children[neighbor] != float('inf')):

    # print(graph)
    return new_obstacle


def moveAndRescan(graph, queue, s_current, scan_range, k_m):
    if(s_current == graph.goal):
        return 'goal', k_m
    else:
        s_last = s_current
        s_new = nextInShortestPath(graph, s_current)
        new_coords = stateNameToCoords(s_new)

        if(graph.cells[new_coords[1]][new_coords[0]] == -1):  # just ran into new obstacle
            s_new = s_current  # need to hold tight and scan/replan first

        results = scanForObstacles(graph, queue, s_new, scan_range, k_m)
        # print(graph)
        k_m += heuristic_from_s(graph, s_last, s_new)
        computeShortestPath(graph, queue, s_current, k_m)

        return s_new, k_m


def initDStarLite(graph, queue, s_start, s_goal, k_m):
    graph.graph[s_goal].rhs = 0
    heapq.heappush(queue, calculateKey(
        graph, s_goal, s_start, k_m) + (s_goal,))
    computeShortestPath(graph, queue, s_start, k_m)

    return (graph, queue, k_m)