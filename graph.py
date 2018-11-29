import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE, PATH_SOLUTION_FILE

nb_dist = 0


# def real_cost(sol):
#     s = 0
#     for i in range(sol.len):
#         s += sol.dist(i,i+1)
#     return s

class Vertex:
    def __init__(self, id x, y):
        self.id = id
        self._x = x
        self._y = y
    def __str__(self):
        return '<{}, {}, {}>'.format(self.id, self._x, self._y)
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def next_vertex(self):
        return self._next_vertex

    def dist_euclid(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Terminal(Vertex):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)

    def isTerminal(self):
        return True

    def isDistrib(self):
        return False

class Distrib(Vertex):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)

    def isTerminal(self):
        return False

    def isDistrib(self):
        return True

class Graph:
    def __init__(self):

        self.vertex = dict()
        self.edges = dict()

        # Parsing graph
        Nodes = getNodes(file_name)
        for id, x, y, code in Nodes:
            if code == 'terminal':
                vertex[id] =  Terminal(id, x, y)
            elif code == 'distribution':
                vertex[id] =  Distrib(id, x, y)

        #Parsing edges
        self.edges = getEdges()

    def __getitem__(self, key):
        return self.vertex[key]

    # def update_distance_dict(self):
    #     for i in range(self.nb_vertex):
    #         for j in range(i):
    #             self._distances[i, j] = self._vertex[i].dist(self._vertex[j])
    #
    #     for j in range(self.nb_vertex):
    #         self._distances[self.nb_vertex, j] = self._vertex[0].dist(self._vertex[j])


    # def dist(self, i, j):
    #     if i > j:
    #         return self._distances[i,j]
    #     elif i < j:
    #         return self._distances[j,i]
    #     else:
    #         return 0

    # def randomize(self, nb_vertex):
    #     self._nb_vertex = nb_vertex
    #     for id in range(nb_vertex):
    #         x = random.random()*self.width
    #         y = random.random()*self.height
    #         self._vertex.append(Vertex(x, y))
    #     self.update_distance_dict()


class Solution:
    def __init__(self, graph):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.loops = [[]]
        self.chains = [[]]

    def __str__(self):
        string = ''
        for id in self._path_index:
            string += self.vertex[id].__str__() + '\n'
        return string

    def __copy__(self):
        return Solution(self.graph)

    def cost_edge(self, id1, id2):
        return self.graph.edges[id1, id2]

    def cost_loop(self, loop):
        pass

    def cost_chain(self, chain):
        pass

    def cost(self):
        pass

    def disturb(self):
        pass

    def write():
        suffix = ""
        file_already_exists = True
        while file_already_exists:
            try:
                fichier = open(PATH_SOLUTION_FILE, 'r')
            except:
                suffix = "1"
                PATH_SOLUTION_FILE = PATH_SOLUTION_FILE[:-3] + "suffix" + ".txt"
        for loop in self.loops:
            line = "b"
            for id in loop:
                line += " " + str(id)
            line += "\n"
            fichier.write(line)
        for chain in self.chains:
            line = "c"
            for id in chain:
                line += " " + str(id)
            line += "\n"
            fichier.write(line)
        fichier.close()


if __name__ == '__main__':
    # g = Graph()
    # g.randomize(1000)
    # V = Vertex(2, 4)
    # print(V)
    # #g.display()
    # list_of_vertex = []
    # for i in range(g.nb_vertex):
    #     list_of_vertex.append(g[i])
    # sol = Solution(g)
    # print(sol)
    # print(sol.cost())
    # sol2 = sol.disturb()
    # print(sol2.cost())
    # sol3 = sol2.disturb()
    # print(sol3.cost())
    # sol4 = sol3.disturb()
    # print(sol4.cost())
