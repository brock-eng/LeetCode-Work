from operator import truediv
from xmlrpc.client import MAXINT

class Graph:
    
    # implementation of lazy dijstra algorithm
    # input format [node_start, node_finish, path_weight]
    def LazyDijkstra(self, edges: list[list], numNodes: int, start:int = 0, end:int = -1) -> list:
        if end == -1: end = numNodes - 1
        
        # Init distance and previous path dicts
        dist, path = dict(), dict()
        for n in range(numNodes):
            dist[n], path[n] = MAXINT, None
        dist[start], path[0] = 0, "Start"

        # Convert edge list to adjacency list
        adjList = [[] for _ in range(numNodes)]
        for edge in edges:
            adjList[edge[0]].append([edge[1], edge[2]])

        # Init a priority queue
        pq = []
        pq.append((0, 0))

        # While priority queue is not empty
        while len(pq) != 0:
            index, minValue = pq.pop(0)
            if dist[index] < minValue: continue
            for edge in adjList[index]:
                newDist = dist[index] + edge[1]
                if newDist < dist[edge[0]]:
                    dist[edge[0]] = newDist
                    path[edge[0]] = index
                    pq.append((edge[0], newDist))

        # Recreate the shortest path from prev array
        prev = []
        last = path[end]
        if not last: return [-1]
        while last != "Start":
            prev.append(last)
            last = path[last]
        prev.reverse()
        prev.append(end)
        prev.append("Distance: " + str(dist[end]))
        return prev

    # implementation of bellmanford algorithm
    # detects negative cycles in a weighted graph
    def BellmanFord(self, edges: list[list], numNodes: int) -> list:
        dist = [MAXINT] * numNodes
        dist[0] = 0
        for _ in range(numNodes - 1):
            for edge in edges:
                if (dist[edge[0]] + edge[2] < dist[edge[1]]):
                    dist[edge[1]] = dist[edge[0]] + edge[2]

        for _ in range(numNodes - 1):
            for edge in edges:
                if (dist[edge[0]] + edge[2] < dist[edge[1]]):
                    dist[edge[1]] = -MAXINT

        return dist

    # find bridges in a graph
    def FindBridges(self, edges: list[list], numNodes: int) -> list:
        adjList = [[] for _ in range(numNodes)]
        for edge in edges:
            adjList[edge[0]].append(edge[1])
            adjList[edge[1]].append(edge[0])

        ids = [0] * numNodes
        low = [0] * numNodes
        visited = [False] * numNodes
        bridges = []

        def dfs(n, pn, id):
            visited[n] = True
            id += 1
            low[n] = ids[n] = id

            for to in adjList[n]:
                if to == pn: continue
                if not visited[to]:
                    dfs(to, n, id)
                    low[n] = min(low[n], low[to])
                    if ids[n] < low[to]:
                        bridges.append([n, to])
                else:
                    low[n] = min(low[n], ids[to])

        for i in range(numNodes):
            if not visited[i]:
                dfs(i, -1, 0)
        
        return bridges
