from sys import maxsize
from itertools import permutations

V = 4 #Number of nodes(destinations to reach)

#Implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s):
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    min_path = maxsize # fetches the largest value a variable of data type Py_ssize_t can store
    next_permutation=permutations(vertex)

    #Algorithm
    for i in next_permutation:
        # store current Path weight(cost)
        current_pathweight = 0
        
        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
        min_path = min(min_path, current_pathweight)

    return min_path

# Driver Code
if __name__ == "__main__":
    # Distance from each node to other nodes stored as matrix
    graph = [[0, 10, 15, 20], [20, 0, 35, 50],
            [15, 35, 0, 30], [20, 25, 30, 0]]
    s = 0 #index of starting node
    print(travellingSalesmanProblem(graph, s))