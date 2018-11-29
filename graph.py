import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE, PATH_SOLUTION_FILE
from parser import *

class Vertex:
    def __init__(self, id, x, y):
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
        Nodes = getNodes()
        for id, x, y, code in Nodes:
            if code == 'terminal':
                self.vertex[id] =  Terminal(id, x, y)
            elif code == 'distribution':
                self.vertex[id] =  Distrib(id, x, y)

        #Parsing edges
        self.edges = getEdges()

    def __getitem__(self, key):
        return self.vertex[key]

class Solution:
    def __init__(self, graph):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.loops = [[]]
        self.chains = [[]]

        for id, value in self.graph.vertex.items():
            self.loops[0].append(id)

    def __copy__(self):
        return Solution(self.graph)

    def cost_edge(self, id1, id2):
        return self.graph.edges[id1, id2]

    def cost_loop(self, loop):
        '''Compute the cost of a given loop'''
        if loop == []:
            return 0

        cost = self.cost_edge(loop[-1], loop[0])

        for i in range(len(loop)-1):
            cost += self.cost_edge(loop[i], loop[i+1])

        return cost

    def cost_chain(self, chain):
        '''Compute the cost of a given chain'''
        if chain == []:
            return 0

        cost = 0
        for i in range(len(chain)-1):
            cost += self.cost_edge(chain[i], chain[i+1])

        return cost

    def cost(self):
        cost = 0
        for loop in self.loops:
            cost += self.cost_loop(loop)
        for chain in self.chains:
            cost == self.cost_chain(loop)

        return cost

    def swap(self):
        idLoop = random.randint(0, len(self.loops)-1)
        i = random.randint(0, len(self.loops[idLoop])-1)
        j = random.randint(0, len(self.loops[idLoop])-1)
        self.loops[idLoop][i], self.loops[idLoop][j] = self.loops[idLoop][j], self.loops[idLoop][i]

    def getRandomIdLoop(self):
        idLoop = random.randint(0, len(self.loops)-1)
        return idLoop

    def disturb(self):
        loop = self.getRandomIdLoop()
        i = random.randint(0, len(loop)-1)
        j = random.randint(0, len(loop)-1)
        return self.reverse(loop, i, j)

    def reverse(self, loop, i, j):
        n = len(loop)
        if i>=n or j>=n or i<0 or j<0:
            raise IndexError("Indice en dehors des bornes")

        i, j = min(i,j), max(i,j)
        if i == 0 and j == n-1:
            return None # Formule du cout non valide dans ce cas la
        if j-i > n-(j-i):
            i, j = j+1, i+n-1

        for k in range((j+1-i)//2):
            i1, i2 = (i+k)%n, (j-k)%n
            loop[i1], loop[i2] = loop[i2], loop[i1]

        return loop

    def write(self):
        file_already_exists = True
        fichier = open(PATH_SOLUTION_FILE, 'w')
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
    g = Graph()
    sol = Solution(g)
    print("cost : {}".format(sol.cost()))
    loop = sol.loops[0]
    print(loop)
    sol.reverse(loop, 2, 5)
    print(loop)
    sol.write()
