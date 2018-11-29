import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE

nb_dist = 0


def real_cost(sol):
    s = 0
    for i in range(sol.len):
        s += sol.dist(i,i+1)
    return s

class Vertex:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    def __str__(self):
        return '<{}, {}>'.format(self._x, self._y)
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def next_vertex(self):
        return self._next_vertex


    def dist(self, other):
        global nb_dist
        nb_dist = nb_dist + 1
        # print(nb_dist)
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Graph:
    def __init__(self, width=1, height=1):
        # self._nb_vertex = nb_vertex
        self._width = width
        self._height = height
        self._dist = dict()
        self._vertex = list()
        self._nb_vertex = 0
        self._distances = dict()


    @property
    def width(self):
        return self._width
    @property
    def nb_vertex(self):
        return self._nb_vertex
    @property
    def height(self):
        return self._height

    def __getitem__(self, key):
        # idx = key
        # if idx in self._vertex:
        return self._vertex[key]
        # else:
        #     raise ValueError("vertex does not exist")

    def update_distance_dict(self):
        for i in range(self.nb_vertex):
            for j in range(i):
                self._distances[i, j] = self._vertex[i].dist(self._vertex[j])

        for j in range(self.nb_vertex):
            self._distances[self.nb_vertex, j] = self._vertex[0].dist(self._vertex[j])


    def dist(self, i, j):
        if i > j:
            return self._distances[i,j]
        elif i < j:
            return self._distances[j,i]
        else:
            return 0

    def randomize(self, nb_vertex):
        self._nb_vertex = nb_vertex
        for id in range(nb_vertex):
            x = random.random()*self.width
            y = random.random()*self.height
            self._vertex.append(Vertex(x, y))
        self.update_distance_dict()




    def display(self, save=False):
        X = np.zeros(self.nb_vertex)
        Y = np.zeros(self.nb_vertex)

        for id in range(self.nb_vertex):
            X[id] = self[id].x
            Y[id] = self[id].y
        plt.scatter(X, Y)
        if save==True:
            plt.savefig(PATH_REFERENCE_GRAPH_FIGURE)
        plt.show()

    def get_nearest_vertex(id_vertex):
        pass

    def get_reference(self):

        """ updates the graph with the values of the testing graph"""

        file = open(PATH_REFERENCE_GRAPH, "r")
        Vertices = []
        for line in file:
            AuxList = []
            Coordinates = line.strip().split('\t')
            for coordinate in Coordinates:
                AuxList.append(float(coordinate))
            Vertices.append(copy.deepcopy(AuxList))
        file.close()
        self._nb_vertex = len(Vertices)
        for i in range(self.nb_vertex):
            vertex = Vertices[i]
            self._vertex.append(Vertex(vertex[0], vertex[1]))
        self.update_distance_dict()

class Solution:
    def __init__(self, graph, path_index = None, cost = None):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.vertex = self.graph._vertex
        self.len = len(self.vertex)

        self._path_index = path_index
        if path_index == None:
            self._path_index = list(range(self.len))


        if cost != None:
            self._current_cost = cost
            self._is_cost_actualized = True
        else:
            self._is_cost_actualized = False
            self._current_cost = self.cost()


    def __getitem__(self, key):
        # if(key > self.len):
        #     raise IndexError()

        if(key == self.len):
            return self.vertex[self._path_index[0]]
        # if(key == -1):
        #     return self.vertex[self._path_index[-1]]
        return self.vertex[self._path_index[key]]

    def __setitem__(self, key, val):
        self.vertex[key] = val

    def __str__(self):
        string = ''
        for id in self._path_index:
            string += self.vertex[id].__str__() + '\n'
        return string

    def __copy__(self):

        return Solution(self.graph, self._path_index[:], self._current_cost if self._is_cost_actualized else None)

    def get_most_distant_vertices_id(self):
        max_dist = -1
        i_max = None
        for i in range(1, self.len):
            new_dist = self[i].dist(self[i+1])
            if new_dist > max_dist:
                max_dist = new_dist
                i_max = i
        return i_max

    def dist(self, i, j):
        return self.graph.dist(self._path_index[i%self.len], self._path_index[j%self.len])

    def set_path_index(self, key, val):
        self._is_cost_actualized = False
        self._path_index[key] = val

    def set_cost(self, cost):
        self._current_cost = cost
        self._is_cost_actualized = True

    def swap(self, i, j):
        # self._is_cost_actualized = False
        new_cost = self._current_cost - self.dist(i, i-1) - self.dist(j, j-1) - self.dist(j, j+1) - self.dist(i, i+1) + self.dist(i-1,j) + self.dist(j, i+1) + self.dist(j-1, i) + self.dist(i, j+1)
        self._path_index[i], self._path_index[j] = self._path_index[j], self._path_index[i]
        self.set_cost(new_cost)
        # self[i], self[j] = self[j], self[i]

    def reverse(self, i, j):
        if i>=self.len or j>=self.len or i<0 or j<0:
            raise IndexError("Indice en dehors des bornes")

        i, j = min(i,j), max(i,j)
        if i == 0 and j == self.len-1:
            return None # Formule du cout non valide dans ce cas la
        if j-i > self.len-(j-i):
            i, j = j+1, i+self.len-1

        i0, j0 = i%self.len, j%self.len
        new_cost = self._current_cost-(self.dist(i0,i0-1)+self.dist(j0,j0+1)-self.dist(i0-1,j0)-self.dist(j0+1,i0))

        for k in range((j+1-i)//2):
            i1, i2 = (i+k)%self.len, (j-k)%self.len
            self._path_index[i1], self._path_index[i2] = self._path_index[i2], self._path_index[i1]

        self.set_cost(new_cost)

    def get_edges_dist(self):
        Dist = []
        for i in range(self.len):
            Dist.append(self.dist(i,i+1))
        return Dist


    def cost(self):

        if self._is_cost_actualized:
            # print("{}".format(real_cost(self)-self._current_cost))
            return self._current_cost

        print("calcul complet")

        s = 0
        for i in range(len(self.vertex)):
            s += self.dist(i,i+1)

        self._current_cost = s
        self._is_cost_actualized = True
        return s

    def disturb(self):
        return self.disturb_reverse()

    def disturb_reverse(self):
        s2 = copy.copy(self)
        id1 = random.randint(0, self.len-1)
        id2 = random.randint(0, self.len-1)
        s2.reverse(id1, id2)
        return s2

    def randomize_solution(self, grid):
        for i in range(grid.nb_vertex):
            self[i] = grid[i]


if __name__ == '__main__':
    g = Graph()
    g.randomize(1000)
    V = Vertex(2, 4)
    print(V)
    #g.display()
    list_of_vertex = []
    for i in range(g.nb_vertex):
        list_of_vertex.append(g[i])
    sol = Solution(g)
    print(sol)
    print(sol.cost())
    sol2 = sol.disturb()
    print(sol2.cost())
    sol3 = sol2.disturb()
    print(sol3.cost())
    sol4 = sol3.disturb()
    print(sol4.cost())
