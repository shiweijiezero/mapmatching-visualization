from typing import List, Tuple, Set, Dict
from .vertex import Vertex

#图
class Graph:
    def __init__(self, *vertices: Vertex):
        length = len(vertices)
        self.matrix = dict()
        for i in range(length):
            self.matrix[vertices[i]] = dict()
            for j in range(length):
                self.matrix[vertices[i]][vertices[j]] = None

    def add_one_edge(self,v1:Vertex,v2:Vertex):
        self.matrix[v1][v2] = self.matrix[v2][v1] = "unvisited"

    def set_visited(self,v1:Vertex,v2:Vertex):
        self.matrix[v1][v2] = self.matrix[v2][v1] = "visited"
