from typing import List, Tuple, Set, Dict
from .vertex import Vertex

#边
class Edge:
    def __init__(self, start_vertex: Vertex, end_vertex: Vertex):
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex

    def __str__(self):
        return "{}\t===>>\t{}".format(self.start_vertex.__str__(), self.end_vertex.__str__())
