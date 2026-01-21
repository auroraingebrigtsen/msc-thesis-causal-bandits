from collections import deque


def topological_sort(V, directed_edges):
    """ # Function to perform Kahn's Algorithm (somewhat stolen from geeksfromgeeks)"""
    parents = {node: set() for node in V}
    for u, v in directed_edges:
        parents[v].add(u)
    in_degree = {node: len(parents) for node, parents in parents.items()}

    children = {node: [] for node in V}
    for node, parents in parents.items():
        for parent in parents:
            children[parent].append(node)

    # Queue for vertices with 0 in-degree
    queue = deque([i for i in list(V) if in_degree[i] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)

        # Decrease in-degree for adjacent vertices
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return topo_order