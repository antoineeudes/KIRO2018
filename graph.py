import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, ceil
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE, PATH_SOLUTION_FILE
from parser import *


def distance2_euclidienne(terminal1_x, terminal1_y, terminal2_x, terminal2_y):
    return (terminal1_x - terminal2_x)**2 + (terminal1_y - terminal2_y)**2

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

        self.id_terminals = []
        self.id_distribs = []
        self.vertex = dict()
        self.edges = dict()

        # Parsing graph
        Nodes = getNodes()
        for id, x, y, code in Nodes:
            if code == 'terminal':
                self.vertex[id] = Terminal(id, float(x), float(y))
                self.id_terminals.append(id)
            elif code == 'distribution':
                self.vertex[id] = Distrib(id, float(x), float(y))
                self.id_distribs.append(id)

        #Parsing edges
        self.edges = getEdges()

    def __getitem__(self, id):
        return self.vertex[id]

def getMedian(L, graph):
    # try:
    #     len(L)
    # except:
    #     L = [L]
    #     print(L)
    s_x = 0
    s_y = 0
    nb_points = len(L)

    for id in L:
        s_x += graph.vertex[id].x
        s_y += graph.vertex[id].y

    return [float(s_x)/nb_points, float(s_y)/nb_points]

def getMediansFromClusters(clusters, graph):
    medians = []
    for cluster in clusters:
        medians.append(getMedian(cluster, graph))
    return medians

def getCluster(medians, terminal):
    id_cluster = 0
    distance_mini = float('inf')
    #print("len medians", len(medians))

    for i in range(len(medians)):
        #print("iiiii = ", i)
        distance = distance2_euclidienne(medians[i][0], medians[i][1], terminal.x, terminal.y)
        if distance<distance_mini:
            #print('chibre')
            distance_mini = distance
            id_cluster = i
            #print('id_cluster', i)

    return id_cluster


def getClustersFromMedians(medians, id_terminals, graph):
    #print("len medians = ", len(medians))
    clusters = []
    for i in range(len(medians)):
        clusters.append([0])
    #print(len(clusters))

    for id in id_terminals:
        #print("id cluster", getCluster(medians, graph.vertex[id]))
        #print("len cluster", len(clusters))
        clusters[getCluster(medians, graph.vertex[id])].append(id)
    return clusters


class Solution:
    def __init__(self, graph, loops=None, chains=None):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.loops = []
        self.chains = [] # Contient des tupple (id, []) ou id est l'id de la boucle a laquelle la classe est ratachee

        self.nbLoops = ceil(float(len(self.graph.vertex))/30)
        # print("nbLoops", self.nbLoops)
        # self.clusterize_distributions(self.nbLoops)
        for i in range(self.nbLoops):
            self.loops.append([])

        if loops == None:
            for id, value in self.graph.vertex.items():
                self.loops[id//30].append(id)

        if loops != None:
            self.loops = loops
        if chains != None:
            self.chains = chains


    def __copy__(self):
        return Solution(self.graph, copy.deepcopy(self.loops), copy.deepcopy(self.chains))

    # def clusterize_distributions(self, nbClusters):
    #     L = []
    #     for i in range(nbClusters):
    #         L.append([])
    #     # print(L)
    #
    #     medians = nbClusters * [0]
    #     nbDistrib = len(self.graph.id_terminals)
    #     #print(nbDistrib)
    #     for i in range(nbDistrib):
    #         L[i//30].append(self.graph.id_terminals[i])
    #         # print(L)
    #         #print(len(medians))
    #
    #     for i in range(len(L)):
    #         print(L[i])
    #
    #     for i in range(len(L)):
    #
    #         #print(L, i)
    #         medians[i] = getMedian(L[i], self.graph)
    #
    #     for i in range(100):
    #         print(i)
    #
    #         L = getClustersFromMedians(medians, self.graph.id_terminals, self.graph)
    #         medians = getMediansFromClusters(L, self.graph)
    #     self.loops = L
    #
    #     for ()



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
        for i in range(len(chain[1])-1):
            cost += self.cost_edge(chain[1][i], chain[1][i+1])

        return cost

    def cost(self):
        cost = 0
        for loop in self.loops:
            cost += self.cost_loop(loop)
        for chain in self.chains:
            cost += self.cost_chain(chain)

        return cost

    def swap(self):
        idLoop = random.randint(0, len(self.loops)-1)
        i = random.randint(0, len(self.loops[idLoop])-1)
        j = random.randint(0, len(self.loops[idLoop])-1)
        self.loops[idLoop][i], self.loops[idLoop][j] = self.loops[idLoop][j], self.loops[idLoop][i]

    def getRandomIdLoop(self):
        idLoop = random.randint(0, len(self.loops)-1)
        return idLoop

    def disturb_in_loop(self):
        idLoop = self.getRandomIdLoop()
        i = random.randint(0, len(self.loops[idLoop])-1)
        j = random.randint(0, len(self.loops[idLoop])-1)
        new_solution = self.reverse(idLoop, i, j)
        if not self.is_loop_admissible(self.loops[idLoop]):
            return self.reverse(idLoop, i, j)
        return new_solution

    def disturb_between_loops(self):
        idLoop1 = self.getRandomIdLoop()
        idLoop2 = self.getRandomIdLoop()
        i = random.randint(0, len(self.loops[idLoop1])-1)
        j = random.randint(0, len(self.loops[idLoop2])-1)
        self.loops[idLoop1][i],  self.loops[idLoop1][j] = self.loops[idLoop1][j], self.loops[idLoop1][i]
        if not (self.is_loop_admissible(self.loops[idLoop1]) and self.is_loop_admissible(self.loops[idLoop2])):
            self.loops[idLoop1][i],  self.loops[idLoop1][j] = self.loops[idLoop1][j], self.loops[idLoop1][i]
            print("pas_pris")
            return self
        return self


    def disturb(self):
        r = random.random()
        if r<1:
            return self.disturb_in_loop()
        else:
            return self.disturb_between_loops()

    def reverse(self, idLoop, i, j):
        n = len(self.loops[idLoop])
        if i>=n or j>=n or i<0 or j<0:
            raise IndexError("Indice en dehors des bornes")

        i, j = min(i,j), max(i,j)

        if j-i > n-(j-i):
            i, j = j+1, i+n-1

        for k in range((j+1-i)//2):
            i1, i2 = (i+k)%n, (j-k)%n
            self.loops[idLoop][i1], self.loops[idLoop][i2] = self.loops[idLoop][i2], self.loops[idLoop][i1]

        return self

    def is_loop_admissible(self, loop):
        if loop == []:
            return True

        nb_distribs = 0
        nb_terminals = 0

        for id in loop:
            if self.graph[id].isDistrib():
                nb_distribs += 1
            elif self.graph[id].isTerminal():
                nb_terminals += 1

        # print("distrib {} terminals {}".format(nb_distribs, nb_terminals))
        return nb_distribs >= 1 and nb_terminals <= 30

    def is_chain_admissible(self, chain):
        if chain == [] or chain[1] == []:
            return True

        id_parent_loop = chain[0]
        chain_elements = chain[1]

        n = len(chain_elements)
        if n > 6:
            return False

        # print(chain_elements)

        # Premier element est dans la boucle a laquelle la chiane appartient
        if not chain_elements[0] in self.loops[id_parent_loop]:
            return False

        for i in range(1, n):
            if chain_elements[i] in self.loops[id_parent_loop]:
                return False

        return True

    def all_terminals_are_joined(self):
        Seen = dict()

        for id_terminal in self.graph.id_terminals:
            Seen[id_terminal] = False

        for loop in self.loops:
            for id_vertex in loop:
                Seen[id_vertex] = True

        for id_loop, chain in self.chains:
            for id_vertex in chain:
                Seen[id_vertex] = True

        for key, seen in Seen.items():
            if not seen:
                return False

        return True

    def isAdmissible(self):
        for loop in self.loops:
            if not self.is_loop_admissible(loop):
                return False
        for chain in self.chains:
            if not self.is_chain_admissible(chain):
                return False

        return self.all_terminals_are_joined()

    def init_random_admissible(self):
        nb_distribs = len(self.graph.id_distribs)
        nb_terminals = len(self.graph.id_terminals)

        loops = [[] for k in range(nb_distribs)]

        nb_terminals_added = 0
        # Add all distribs in different loops
        for i in range(nb_distribs):
            loops[i].append(self.graph.id_distribs[i])

            k = 0
            while k<30 and nb_terminals_added < nb_terminals:
                loops[i].append(self.graph.id_terminals[nb_terminals_added])
                nb_terminals_added += 1
                k += 1

        self.loops = loops

        #If remaining non affected terminals
        #we have to create chains
        nb_chains_to_create = ceil((nb_terminals-nb_terminals_added)/5.)
        chains = [[-1, []] for i in range(nb_chains_to_create)]

        #Attributing chains to loops
        for i in range(len(chains)):
            id_loop = random.randint(0, len(self.loops)-1)
            id_vertex = random.randint(0, len(self.loops[id_loop])-1)
            # print(chains[i])
            # print(chains[i][0])
            # print(self.loops[id_loop][id_vertex])
            chains[i][0] = id_loop#self.loops[id_loop][id_vertex]
            # print(id_loop)
            # print(id_vertex)

            # print(self.loops[id_loop][id_vertex])
            chains[i][1].append(self.loops[id_loop][id_vertex])
            # print(chains[i][0])
            # print(id_loop)

        k = 0
        id_chain = 0
        while nb_terminals_added < nb_terminals:
                if k >= 5:
                    k=0
                    id_chain += 1
                chains[id_chain][1].append(self.graph.id_terminals[nb_terminals_added])
                k += 1
                nb_terminals_added += 1


        #To do


        self.chains = chains

        # print("CHAINS")
        # print(chains)






    def write(self):
        for loop in self.loops:
            while not self.graph.vertex[loop[0]].isDistrib():
                loop.append(loop[0])
                del(loop[0])
        fichier = open(PATH_SOLUTION_FILE, 'w')
        for loop in self.loops:
            if loop == []:
                continue
            line = "b"
            for id in loop:
                line += " " + str(id)
            line += "\n"
            fichier.write(line)
        for id_loop, chain in self.chains:
            if chain == [] or chain[1] == []:
                continue
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
    #print(loop)
    sol.reverse(0, 2, 5)
    # print(loop)
    # print(sol.graph.id_distribs)
    # print(sol.graph.id_terminals)
    #
    # print(sol.loops)
    # print(sol.chains)

    # if sol.isAdmissible():
    #     print("Admissible")
    # else:
    #     print("non admissible")

    sol.init_random_admissible()
    print(sol.loops)
    print(sol.chains)
    print(sol.isAdmissible())
    print(sol.cost())

    sol.write()

    # g = Graph()
    # sol = Solution(g)
    # print("cost : {}".format(sol.cost()))
    # sol.clusterize_distributions(5)
    # print(sol.id.id_distribs)
    # print(sol.id.id_terminals)
    # sol.write()
