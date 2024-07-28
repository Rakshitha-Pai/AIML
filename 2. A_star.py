def aStarAlgo(start_node, stop_node, Graph_nodes):
    open_set = {start_node}
    closed_set = set()
    g = {start_node: 0}
    parents = {}
    
    while open_set:
        n = None
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] is None:
            break
        open_set.remove(n)
        closed_set.add(n)
        for (m, weight) in get_neighbors(n, Graph_nodes):
            if m not in closed_set:
                if m not in open_set or g[m] > g[n] + weight:
                    parents[m] = n
                    g[m] = g[n] + weight
                    open_set.add(m)

    if n is None:
        print('Path does not exist!')
        return None

    path = []
    while n is not None:
        path.append(n)
        n = parents.get(n)
    return path[::-1]

def get_neighbors(v, Graph_nodes):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

def heuristic(n):
    H_dist = {
        'A': 11,
        'B': 6,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist[n]

Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('A', 6), ('C', 3), ('D', 2)],
    'C': [('B', 3), ('D', 1), ('E', 5)],
    'D': [('B', 2), ('C', 1), ('E', 8)],
    'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)],
    'F': [('A', 3), ('G', 1), ('H', 7)],
    'G': [('F', 1), ('I', 3)],
    'H': [('F', 7), ('I', 2)],
    'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)],
}

print(aStarAlgo('A', 'J', Graph_nodes))





#astar
def astar(start, stop):
    open_set, closed_set = {start}, set()
    g, parents = {start: 0}, {start: start}
    heuristic = lambda n: {'A': 10, 'B': 8, 'C': 5, 'D': 7, 'E': 3, 'F': 6, 'G': 5, 'H': 4, 'I': 1, 'J': 0}[n]

    while open_set:
        n = min(open_set, key=lambda v: g[v] + heuristic(v))

        if n == stop:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start)
            path.reverse()
            print('Path found:', path)
            return path

        open_set.remove(n)
        closed_set.add(n)

        for m, weight in graph_nodes.get(n, []):
            if m in closed_set:
                continue

            tentative_g = g[n] + weight

            if m not in open_set or tentative_g < g[m]:
                g[m], parents[m] = tentative_g, n
                open_set.add(m)

    print('Path doesn\'t exist')
    return None

graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)]
}

astar('A', 'J')


# Path found: ['A', 'F', 'G', 'I', 'J']
# ['A', 'F', 'G', 'I', 'J']
