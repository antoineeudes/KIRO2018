import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, ceil
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE, PATH_SOLUTION_FILE
from parser import *

plt.ion()

def closest_point_in_list(idPoint, listIds, graph):
    distance_mini = float('inf')
    ind_mini = 0
    for id in listIds:
        distance = graph.edges[idPoint, id]
        if distance < distance_mini:
            distance_mini = distance
            ind_mini = id

    return id


def areDifferent(medians, new_medians):
    n = len(medians)
    for i in range(n):
        if medians[i][0]!=new_medians[i][0] or medians[i][1]!=new_medians[i][1]:
            return True
    return False


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

def getCluster(medians, vertex):
    id_cluster = 0
    distance_mini = float('inf')
    #print("len medians", len(medians))

    for i in range(len(medians)):
        #print("iiiii = ", i)
        distance = distance2_euclidienne(medians[i][0], medians[i][1], vertex.x, vertex.y)
        if distance<distance_mini:
            #print('chibre')
            distance_mini = distance
            id_cluster = i
            #print('id_cluster', i)

    return id_cluster

def getClusterFromDistribs(distribs, vertex):
    id_cluster = 0
    distance_mini = float('inf')
    #print("len medians", len(medians))

    for i in range(len(distribs)):
        distrib = distribs[i]
        #print("iiiii = ", i)
        distance = distance2_euclidienne(distrib.x, distrib.y, vertex.x, vertex.y)
        if distance<distance_mini:
            #print('chibre')
            distance_mini = distance
            id_cluster = i
            #print('id_cluster', i)

    return id_cluster


def getClustersFromMedians(medians, graph):
    #print("len medians = ", len(medians))
    clusters = []
    for i in range(len(medians)):
        clusters.append([])
    #print(len(clusters))

    for id, vertex in graph.vertex.items():
        clusters[getCluster(medians, vertex)].append(id)
    return clusters

def getClustersFromDistribs(distribs, graph):
    #print("len medians = ", len(medians))
    clusters = []
    for i in range(len(distribs)):
        clusters.append([])
    #print(len(clusters))

    for id, vertex in graph.vertex.items():
        clusters[getClusterFromDistribs(distribs, vertex)].append(id)
    return clusters

class Loop: # Represente une unique boucle
    def __init__(self, elements_id = [], loop_chains = []):
        self.elements_id = elements_id # Contient les id des vertex composant la boucle
        # self.chains_dict = chains_dict # Dictionnaire associant à un element de la loop une liste d'id representant la chaine
        # self.all_chains = all_chains # Tableau partagé contenant toutes les chaines du graphe
        self.loop_chains = loop_chains

    def create_chain(self, parent_node_id, chain_list):
        '''Créée une chaine a un element de la loop'''
        if parent_node_id in self.elements_id:
            chain = Chain(chain_list, self, parent_node_id)
            # try:
            #     self.chains_dict[parent_node_id].append(chain)
            # except:
            #     self.chains_dict[parent_node_id] = [chain]
            self.loop_chains.append(chain)
            # self.all_chains.append(chain)
        else:
            print("set_chain : element pas dans la loop")

    def add_chain(self, chain):
        '''Ajoute un objet de type Chain a une Loop'''
        # if len(chain.elements_id) == 0:
        #     print("chaine vide supprimée")
        #     return
        parent_node_id = chain.parent_node_id
        if parent_node_id in self.elements_id:
            # try:
            #     self.chains_dict[parent_node_id].append(chain)
            # except:
            #     self.chains_dict[parent_node_id] = [chain]
            chain.parent_loop = self
            self.loop_chains.append(chain)
            # self.all_chains.append(chain)
        else:
            print("set_chain : element pas dans la loop")

    # def get_chains(self, element_rank):
    #     '''Renvoie la liste des chaines accrochees a un sommet'''
    #     return self.chains_dict[element_rank]

    def get_id_elements_with_chain(self):
        '''Renvoie les id de la loop qui ont une chaine'''
        L = []
        for chain in self.loop_chains:
            L.append(chain.parent_node_id)

        return L

    def get_chains_by_id_dict(self):
        '''Renvoie les id de la loop qui ont une chaine'''
        d = dict()
        for chain in self.loop_chains:
            try:
                d[chain.parent_node_id].append(chain)
            except:
                d[chain.parent_node_id] = [chain]

        return d

    def get_list_of_chains(self):
        return self.loop_chains

    def __getitem__(self, key):
        return self.elements_id[key]

    def __setitem__(self, key, value):
        self.elements_id[key] = value

    def __delitem__(self, key):
        del(self.elements_id[key])

    def remove_chain(self, chain_to_del):
        k = 0
        for i in range(len(self.loop_chains)):
            if self.loop_chains[i-k] == chain_to_del:
                del(self.loop_chains[i-k])
                k += 1
                # break

    def get_chains_with_parent_id(self, parent_id):
        if len(self.loop_chains) == 0:
            return []

        L = []
        for chain in self.loop_chains:
            if chain.parent_node_id == parent_id:
                L.append(chain)
        return L

    def getRandomChain(self):
        if len(self.loop_chains) == 0:
            return None

        return self.loop_chains[random.randint(0, len(self.loop_chains)-1)]

        # for i in range(len(self.all_chains)):
        #     if self.all_chains[i] == chain_to_del:
        #         del(self.all_chains[i])
        #         break


        # for parent_id in list(self.chains_dict):
        #     chains = self.chains_dict[parent_id]
        #     k = 0
        #     for i in range(len(chains)):
        #         if chains[i-k] == chain_to_del:
        #             del(chains[i-k])
        #             k += 1
        #     if len(chains) == 0: # Suppresion de la cle si plus de chaine
        #         self.chains_dict.pop(parent_id, None)
        #
        # k = 0;
        # for i in range(len(self.all_chains)):
        #     if self.all_chains[i-k] == chain_to_del:
        #         del(self.all_chains[i-k])
        #         k += 1

class Chain:
    def __init__(self, elements_id = [], parent_loop = None, parent_node_id = None):
        self.elements_id = elements_id
        self.parent_loop = parent_loop
        self.parent_node_id = parent_node_id
        # self.all_chains = all_chains

    def __getitem__(self, key):
        return self.elements_id[key]

    def __setitem__(self, key, value):
        self.elements_id[key] = value

    def __delitem__(self, key):
        del(self.elements_id[key])

    def change_anchor_node(self, new_anchor_id):
        loop = self.parent_loop
        loop.remove_chain(self)
        self.parent_node_id = new_anchor_id
        loop.add_chain(self)

    def delete_from_parent_loop(self):
        # print("parent_id {} {} {}".format(self.parent_node_id, self.elements_id, self.parent_loop))
        if self.parent_loop != None:
            self.parent_loop.remove_chain(self)
        self.parent_node_id = None
        self.parent_loop = None





class Solution:
    def __init__(self, graph, loops=None):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.loops = [] # Contient des objets de type Loop
        # self.all_chains = [] # Contient des objets de type Chain (partagé avec chain et loop)

        if loops != None:
            self.loops = loops
        # if all_chains != None:
        #     self.all_chains = all_chains


    def __copy__(self):
        return Solution(self.graph, copy.deepcopy(self.loops))

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

    def heuristique(self):
        nbVertices = len(self.graph.vertex)
        print(nbVertices)
        nbClusters = min(nbVertices//30 + 1, len(self.graph.id_distribs))
        clusters = []
        print(nbClusters)

        for i in range(nbClusters):
            clusters.append([])

        for id in range(min(30*nbClusters, nbVertices)):
            clusters[id//30].append(id)
        for id in range(30*nbClusters, nbVertices):
            clusters[id%nbClusters].append(id)


        new_medians = getMediansFromClusters(clusters, self.graph)
        medians = nbClusters * [[0, 0]]

        while areDifferent(medians, new_medians):
            medians = copy.deepcopy(new_medians)
            clusters = getClustersFromMedians(new_medians, self.graph)
            new_medians = getMediansFromClusters(clusters, self.graph)
            print("computing")

        # print(clusters)
        s = 0
        for cluster in clusters:
            s+=len(cluster)

        print(s==nbVertices, "GOALLLLLL")


        if nbClusters==len(self.graph.id_distribs):
            available_distribs = []
            nbDistribs = []
            for i in range(nbClusters):
                nbDistrib = 0
                for j in range(len(clusters[i])):
                    if self.graph.vertex[clusters[i][j]].isDistrib():
                        nbDistrib += 1
                        if nbDistrib > 1:
                            available_distribs.append([i, j, clusters[i][j]])
                nbDistribs.append(nbDistrib)
            for i in range(nbClusters):
                if nbDistribs[i]==0:
                    print(available_distribs)
                    clusters[i].insert(available_distribs[-1][-1], 0)
                    del(clusters[available_distribs[-1][0]][available_distribs[-1][1]])
                    del(available_distribs[-1])
        # self.all_chains = []
        self.loops = []
        s = 0
        for cluster in clusters:
            s+=len(cluster)
            self.loops.append(Loop(cluster, []))

        print(s==nbVertices, "GOALLLLLL")

        # self.loops = clusters
        chains = []
        points_in_chains = []
        for i in range(nbClusters):
            n_i = len(clusters[i])
            if n_i > 30:
                points_in_chains = clusters[i][30:]
                nbChainsInLoop = ceil(len(points_in_chains)/4.)
                for k in range(nbChainsInLoop):
                    # chains.append([i, [clusters[i][k]] + clusters[i][30+4*k:30+4*(k+1)]])
                    self.loops[i].create_chain(clusters[i][k], clusters[i][30+4*k:30+4*(k+1)])
                # chains.append([i, [clusters[i][nbChainsInLoop]] + clusters[i][30+4*nbChainsInLoop:]])
                self.loops[i].create_chain(clusters[i][nbChainsInLoop], clusters[i][30+4*nbChainsInLoop:])
                for j in range(30, n_i):
                    del(clusters[i][-1])

        # chains = [[0, [clusters[0][0]] + points_in_chains]]


        for i in range(nbClusters):
            nbDistrib = 0
            for j in range(len(clusters[i])):

                if self.graph.vertex[clusters[i][j]].isDistrib():
                    nbDistrib += 1
            print(nbDistrib)
        print(chains)
        # self.chains = chains

    def heuristique2(self):
        nbVertices = len(self.graph.vertex)
        print(nbVertices)
        nbClusters = len(self.graph.id_distribs)
        clusters = []
        print(nbClusters)

        # for i in range(nbClusters):
        #     clusters.append([])
        #
        # for id in range(min(30*nbClusters, nbVertices)):
        #     clusters[id//30].append(id)
        # for id in range(30*nbClusters, nbVertices):
        #     clusters[id%nbClusters].append(id)
        distribs_list = [self.graph[id_distrib] for id_distrib in self.graph.id_distribs]
        clusters = getClustersFromDistribs(distribs_list, self.graph)


        # new_medians = getMediansFromClusters(clusters, self.graph)
        # medians = nbClusters * [[0, 0]]
        #
        # while areDifferent(medians, new_medians):
        #     medians = copy.deepcopy(new_medians)
        #     clusters = getClustersFromMedians(new_medians, self.graph)
        #     new_medians = getMediansFromClusters(clusters, self.graph)
        #     print("computing")

        # print(clusters)
        # s = 0
        # for cluster in clusters:
        #     s+=len(cluster)
        #
        # print(s==nbVertices, "GOALLLLLL")

        available_distribs = []
        nbDistribs = []
        for i in range(nbClusters):
            nbDistrib = 0
            for j in range(len(clusters[i])):
                if self.graph.vertex[clusters[i][j]].isDistrib():
                    nbDistrib += 1
                    if nbDistrib > 1:
                        available_distribs.append([i, j, clusters[i][j]])
            nbDistribs.append(nbDistrib)
        for i in range(nbClusters):
            if nbDistribs[i]==0:
                print(available_distribs)
                clusters[i].insert(available_distribs[-1][-1], 0)
                del(clusters[available_distribs[-1][0]][available_distribs[-1][1]])
                del(available_distribs[-1])

        # self.all_chains = []
        self.loops = []
        s = 0
        for cluster in clusters:
            s+=len(cluster)
            self.loops.append(Loop(cluster, []))

        print(s==nbVertices, "GOALLLLLL")

        # self.loops = clusters
        chains = []
        points_in_chains = []
        for i in range(nbClusters):
            n_i = len(clusters[i])
            if n_i > 30:
                points_in_chains = clusters[i][30:]
                nbChainsInLoop = ceil(len(points_in_chains)/4.)
                for k in range(nbChainsInLoop):
                    # chains.append([i, [clusters[i][k]] + clusters[i][30+4*k:30+4*(k+1)]])
                    self.loops[i].create_chain(clusters[i][k], clusters[i][30+4*k:30+4*(k+1)])
                # chains.append([i, [clusters[i][nbChainsInLoop]] + clusters[i][30+4*nbChainsInLoop:]])
                self.loops[i].create_chain(clusters[i][nbChainsInLoop], clusters[i][30+4*nbChainsInLoop:])
                for j in range(30, n_i):
                    del(clusters[i][-1])

        # chains = [[0, [clusters[0][0]] + points_in_chains]]


        for i in range(nbClusters):
            nbDistrib = 0
            for j in range(len(clusters[i])):

                if self.graph.vertex[clusters[i][j]].isDistrib():
                    nbDistrib += 1
            print(nbDistrib)
        print(chains)
        # self.chains = chains


    def cost_edge(self, id1, id2):
        return self.graph.edges[id1, id2]

    def cost_loop(self, loop):
        '''Compute the cost of a given loop'''

        if loop == []:
            return 0

        cost = self.cost_edge(loop[-1], loop[0])

        for i in range(len(loop.elements_id)-1):
            cost += self.cost_edge(loop[i], loop[i+1])

        return cost

    def cost_chain(self, chain):
        '''Compute the cost of a given chain'''
        if chain.elements_id == []:
            return 0

        cost = 0
        # print("CHAINE")
        # print(chain)
        cost += self.cost_edge(chain.parent_node_id, chain.elements_id[0])
        for i in range(len(chain.elements_id)-1):
            cost += self.cost_edge(chain.elements_id[i], chain.elements_id[i+1])

        return cost

    def cost(self):
        cost = 0
        for loop in self.loops:
            cost += self.cost_loop(loop)
            # print(loop.get_list_of_chains())
            # for parent_id, chains in loop.chains_dict.items():
            #     for chain in chains:
                    # print("cccccchaine")
                    # print(chain)
            for chain in loop.loop_chains:
                cost += self.cost_chain(chain)

        return cost

    # def swap(self):
    #     idLoop = random.randint(0, len(self.loops)-1)
    #     i = random.randint(0, len(self.loops[idLoop])-1)
    #     j = random.randint(0, len(self.loops[idLoop])-1)
    #     self.loops[idLoop][i], self.loops[idLoop][j] = self.loops[idLoop][j], self.loops[idLoop][i]

    def getRandomIdLoop(self):
        idLoop = random.randint(0, len(self.loops)-1)
        return idLoop

    def getRandomChain(self):
        loop = self.loops[self.getRandomIdLoop()]
        if len(loop.loop_chains) == 0:
            return None
        id = random.randint(0, len(loop.loop_chains)-1)
        return loop.loop_chains[id]

    def disturb_in_loop(self):
        new_solution = copy.copy(self)
        idLoop = self.getRandomIdLoop()
        i = random.randint(0, len(new_solution.loops[idLoop].elements_id)-1)
        j = random.randint(0, len(new_solution.loops[idLoop].elements_id)-1)
        new_solution = new_solution.reverse(idLoop, i, j) #Aucune influence sur les chaines
        if not new_solution.is_loop_admissible(new_solution.loops[idLoop]):
            # print("pas pris")
            return self
        else:
            return new_solution

    def disturb_between_loops(self):
        idLoop1 = self.getRandomIdLoop()
        idLoop2 = self.getRandomIdLoop()

        if idLoop1 == idLoop2:
            return self # Tres important sinon faux : si deux fois la meme loop,
            # une meme chaine peut etre consideree 2 fois et dupliquee

        i = random.randint(0, len(self.loops[idLoop1].elements_id)-1)
        j = random.randint(0, len(self.loops[idLoop2].elements_id)-1)
        if type(self.graph[self.loops[idLoop1][i]]) is type(self.graph[self.loops[idLoop2][j]]):
            # Echange un distrib avec un distrib et un terminal avec un terminal
            # Sinon peut rendre non admissible

            new_solution = copy.copy(self)
            loop1 = new_solution.loops[idLoop1]
            loop2 = new_solution.loops[idLoop2]

            chains_1 = loop1.get_chains_with_parent_id(loop1[i])
            chains_2 = loop2.get_chains_with_parent_id(loop2[j])

            # Mise a jour des attributs des chaines :
            # Retrait des chaines des noeuds parents
            if chains_1 != None:
                for chain1 in chains_1:
                    chain1.delete_from_parent_loop()
            if chains_2 != None:
                for chain2 in chains_2:
                    chain2.delete_from_parent_loop()

            # Echanges des noeuds parents
            loop1[i], loop2[j] = loop2[j], loop1[i]
            # On raccroche les chaines aux nouveaux noeuds parents
            if chains_1 != None:
                for chain1 in chains_1:
                    chain1.parent_node_id = loop2[j]
                    chain1.parent_loop = loop2
                    loop2.add_chain(chain1)

            if chains_2 != None:
                for chain2 in chains_2:
                    chain2.parent_node_id = loop1[i]
                    chain2.parent_loop = loop1
                    loop1.add_chain(chain2)

            return new_solution

        return self

    def disturb_in_chain(self):

        chain = self.getRandomChain()
        if chain == None:
            return self
        new_solution = copy.copy(self)

        if len(chain.elements_id)<=1:
            return self

        i = random.randint(0, len(chain.elements_id)-1)
        j = random.randint(0, len(chain.elements_id)-1)
        chain[i],  chain[j] = chain[j], chain[i]
        # if not (new_solution.is_chain_admissible(chain)):
        #     # print("pas_pris")
        #     return self
        # else:
        return new_solution

    def disturb_remove_from_chain_to_loop(self):
        '''Essai d'enelever un element d'une chaine pour le mettre dans la boucle'''
        idLoop = self.getRandomIdLoop()
        if len(self.loops[idLoop].elements_id) > 30:
            return self #Plus de place

        if len(self.loops[idLoop].loop_chains) == 0:
            return self # Aucune chaine dans la loop

        new_solution = copy.copy(self)
        loop = new_solution.loops[idLoop]
        i_chain = random.randint(0, len(loop.loop_chains)-1)
        chain = loop.loop_chains[i_chain]
        if len(chain.elements_id) == 0:
            return self # chaine choisie vide

        i_chain_element = random.randint(0, len(chain.elements_id)-1)

        pos = random.randint(0, len(loop.elements_id)-1)
        loop.elements_id.insert(pos, chain[i_chain_element]) # Ajout dans la boucle

        if len(chain.elements_id) == 1: # Si on va vider la chaine
            del(loop.loop_chains[i_chain])

        del(chain[i_chain_element]) # Suppression dans la chaine
        return new_solution

    def disturb_create_new_chain(self):
        '''Cree une nouvelle chaine a partir d'un element de la loop'''
        idLoop = self.getRandomIdLoop()
        if len(self.loops[idLoop].elements_id) <= 1:
            return self
        # indice de l'element a enlever
        i = random.randint(0, len(self.loops[idLoop].elements_id)-1)
        if isinstance(self.graph[self.loops[idLoop].elements_id[i]], Distrib):
            return self

        # elements_with_chains = self.loops[idLoop].get_id_elements_with_chain()
        new_solution = copy.copy(self)
        chains_by_id = new_solution.loops[idLoop].get_chains_by_id_dict()
        loop = new_solution.loops[idLoop]
        element_id = loop.elements_id[i]
        # if self.loops[idLoop].elements_id[i] in elements_with_chains:
        if element_id in chains_by_id.keys() and len(chains_by_id[element_id]) >= 2 :
            return self

        p = random.randint(0,1)
        if p==0:
            new_i = i-1
        else:
            new_i = (i+1)%len(loop.elements_id)

        new_parent_id = loop.elements_id[new_i]

        if not element_id in chains_by_id.keys(): # Pas de chaine sur l'element selectionne
            loop.create_chain(new_parent_id, [element_id])
            del(loop.elements_id[i])
            # print("pas de chaine")
            return new_solution
        elif len(chains_by_id[element_id]) == 1: # Une unique chaine partant de l'element selectionne
            # print("une seule chaine")
            chain = chains_by_id[element_id][0]
            if len(chain.elements_id) >= 5: # Plus de place
                return self
            # print("pris")
            chain.elements_id.insert(0, element_id) # Ajout de l'element a la boucle au debut de celle-ci
            chain.parent_node_id = new_parent_id
            # print("new_parent : {}, element : {}".format(new_parent_id, loop.elements_id[i]))
            del(loop.elements_id[i])
            return new_solution

        # del(loop.elements_id[i]) # Suppression de l'element deplace dans la loop
        return new_solution




    def disturb(self):
        # r = random.random()
        # if r<0.5:
        #     return self.disturb_in_loop()

        i = random.randint(0, 8)

        if i == 0:
            return self.disturb_remove_from_chain_to_loop()
        elif i == 1:
            return self.disturb_between_chains()
        elif i == 2:
            return self.disturb_in_chain()
        elif i == 3:
            return self.disturb_anchor_point_in_loop()
        elif i == 4:
            return self.disturb_create_new_chain()
        elif i == 5:
            return self.disturb_anchor_point_in_other_loop()
        elif i == 6:
            return self.disturb_between_loops()
        elif i == 7:
            # j = random.randint(0, 2)
            j = 0
            if j == 0:
                return self.disturb_in_loop()
        elif i >= 8:
            return self.disturb_transfer_from_chain_to_chain()

        return self

    def disturb_between_chains(self):
        # id_loop = self.getRandomIdLoop()
        # if len(self.loops[id_loop].loop_chains) <= 1:
        #     return self # pas assez de chaines dans la boucle
        # new_solution = copy.copy(self)
        # loop = new_solution.loops[id_loop]

        new_solution = copy.copy(self)

        chain1 = new_solution.getRandomChain()
        chain2 = new_solution.getRandomChain()

        if chain1 == None or chain2 == None:
            return self
        if len(chain1.elements_id) <=1 or len(chain2.elements_id) <= 1:
            return self

        id1 = random.randint(1, len(chain1.elements_id)-1)
        id2 = random.randint(1, len(chain2.elements_id)-1)
        chain1[id1], chain2[id2] = chain2[id2], chain1[id1]

        return new_solution

    def disturb_transfer_from_chain_to_chain(self):
        '''Transfere un element de la chaine1 vers la chaine2'''
        id_loop = self.getRandomIdLoop()
        if len(self.loops[id_loop].loop_chains) <= 1:
            return self # Pas assez de chaines

        id_chain1 = random.randint(0, len(self.loops[id_loop].loop_chains)-1)
        id_chain2 = random.randint(0, len(self.loops[id_loop].loop_chains)-1)

        if id_chain1 == id_chain2:
            return self

        if len(self.loops[id_loop].loop_chains[id_chain1].elements_id) == 0:
            return self
        # if len(self.loops[id_loop].loop_chains[id_chain2].elements_id) == 0:
        #     return self
        if len(self.loops[id_loop].loop_chains[id_chain2].elements_id) >= 5:
            return self # Plus de place

        new_solution = copy.copy(self)

        chain1 = new_solution.loops[id_loop].loop_chains[id_chain1]
        chain2 = new_solution.loops[id_loop].loop_chains[id_chain2]

        i1 = random.randint(0, len(chain1.elements_id)-1)
        pos2 = random.randint(0, len(chain2.elements_id))

        # print("{} \nchain 1 {}".format(chain1.elements_id[i1], chain1.elements_id))
        # print("chain 2 {}".format(chain2.elements_id))
        chain2.elements_id.insert(pos2, chain1.elements_id[i1])
        # print("chain 2 {}".format(chain2.elements_id))
        del(chain1.elements_id[i1])
        # print("chain 1 {}".format(chain1.elements_id))

        return new_solution


    def disturb_anchor_point_in_loop(self):
        new_solution = copy.copy(self)
        chain = new_solution.getRandomChain()
        if chain == None:
            return self
        p = random.randint(0, 1)
        i = random.randint(0, len(chain.parent_loop.elements_id)-1)
        if p == 0:
            # On favorise les noeuds juste a cote du point d'ancrage actuel
            # (donc dans la même chaine)
            for j in range(len(chain.parent_loop.elements_id)):
                if chain.parent_loop.elements_id[j] == chain.parent_node_id:
                    # On a retrouve la pos j du point d'ancrage
                    # On ratache a sa droite ou a sa gauche
                    p = random.randint(0, 1)
                    if p == 0:
                        i = j-1
                    else:
                        i = (j+1)%len(chain.parent_loop.elements_id)
                    break

        chain.change_anchor_node(chain.parent_loop.elements_id[i])
        return new_solution

    def disturb_anchor_point_in_other_loop(self):
        new_solution = copy.copy(self)
        chain = new_solution.getRandomChain()
        if chain == None:
            return self
        new_loop = new_solution.loops[new_solution.getRandomIdLoop()]
        chain.delete_from_parent_loop()
        new_parent_id = random.randint(0, len(new_loop.elements_id)-1)
        chain.parent_node_id = new_loop.elements_id[new_parent_id]
        # chain.parent_loop = new_loop
        new_loop.add_chain(chain)

        return new_solution


    def reverse(self, idLoop, i, j):
        n = len(self.loops[idLoop].elements_id)
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
        if loop.elements_id == []:
            return True

        nb_distribs = 0
        nb_terminals = 0

        for id in loop.elements_id:
            if self.graph[id].isDistrib():
                nb_distribs += 1
            elif self.graph[id].isTerminal():
                nb_terminals += 1

        if nb_distribs == 0:
            print("Aucun point de distribution dans la boucle")
            return False

        if nb_terminals > 30:
            print("Plus de 30 antennes dans la boucle")
            return False

        return True

    def is_chain_admissible(self, chain):
        if not chain.parent_node_id in chain.parent_loop.elements_id:
            print("Chaine non admissible car parent_node_id {} pas dans parent_loop : {}".format(chain.parent_node_id, chain.parent_loop.elements_id))
            return False

        if chain.elements_id == []:
            return True

        # "Au plus 5 sommets ne sont pas dans la boucle"
        k = 0
        for id in chain.elements_id:
            if not id in chain.parent_loop.elements_id:
                k += 1
        if k > 5:
            print("Plus de 5 sommets hors boucle dans la chaine : ".format(chain.elements_id))
            return False

        return True

    def all_terminals_are_joined(self):
        Seen = dict()

        for id_terminal in self.graph.id_terminals:
            Seen[id_terminal] = 0
        for id_distrib in self.graph.id_distribs:
            Seen[id_distrib] = 0

        for loop in self.loops:
            for id_vertex in loop.elements_id:
                Seen[id_vertex] += 1

            for chain in loop.loop_chains:
                for id_vertex in chain.elements_id:
                    Seen[id_vertex] += 1

        for key, seen in Seen.items():
            if seen == 0:
                print("Toutes les antennes ne sont pas reliées : {}".format(key))
                return False
            if seen > 1:
                print("Une antenne est reliée plusieurs fois : {} est reliée {} fois".format(key, seen))
                return False

        return True

    def isAdmissible(self):
        for loop in self.loops:
            if not self.is_loop_admissible(loop):
                print("Boucle non admissible")
                return False
            for chain in loop.loop_chains:
                if not self.is_chain_admissible(chain):
                    print("Chaine non admissible : {}".format(chain.elements_id))
                    return False

        # for chain in self.all_chains:
        #     if not self.is_chain_admissible(chain):
        #         print("Chaine non admissible : {}".format(chain.elements_id))
        #         return False

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

    # def prepare(self):
    #
    #     for loop in self.loops:
    #         L = []
    #         for element_id, chain in loop.chains_dict.items():
    #             if len(chain) > 1:
    #                 L.append(chain)
    #         loop.chains_dict[element_id] = L

    def write(self):
        if not self.isAdmissible():
            print("NOT ADMISSIBLE")
            return None
        for loop in self.loops:
            k = 0
            while k<len(loop.elements_id) and not self.graph.vertex[loop[0]].isDistrib():
                loop.elements_id.append(loop[0])
                del(loop[0])
                k += 1
        fichier = open(PATH_SOLUTION_FILE, 'w')
        for loop in self.loops:
            if loop.elements_id == []:
                continue
            line = "b"
            for id in loop.elements_id:
                line += " " + str(id)
            line += "\n"
            fichier.write(line)

            # for parent_id, chains in loop.chains_dict.items():
            #     for chain in chains:
            for chain in loop.loop_chains:
                if chain.elements_id == []:
                    continue
                line = "c" + " " + str(chain.parent_node_id)
                for id in chain.elements_id:
                    line += " " + str(id)
                line += "\n"
                fichier.write(line)
            # for chain in self.all_chains:
            #     if chain.elements_id == []:
            #         continue
            #     line = "c" + " " + str(chain.parent_node_id)
            #     for id in chain.elements_id:
            #         line += " " + str(id)
            #     line += "\n"
            #     fichier.write(line)
        fichier.close()

    def show(self, block=True):
        plt.clf()
        # colors = "bgrcmk"
        # nb_colors = len(colors)

        for i in range(len(self.loops)):
            loop = self.loops[i]
            for j in range(len(loop.elements_id)):
                id_node1 = loop[j-1]
                id_node2 = loop[j]
                x = [self.graph[id_node1].x, self.graph[id_node2].x]
                y = [self.graph[id_node1].y, self.graph[id_node2].y]
                plt.plot(x, y, marker=",", color='black')#colors[i%nb_colors])

            # for parent_node_id, chains in loop.chains_dict.items():
            #     for chain in chains:
            #         # id_node0, chain = self.chains[i]
            for chain in loop.loop_chains:
                parent_node_id = chain.parent_node_id
                if len(chain.elements_id) == 0:
                    continue
                x = [self.graph[parent_node_id].x, self.graph[chain[0]].x]
                y = [self.graph[parent_node_id].y, self.graph[chain[0]].y]
                plt.plot(x, y, marker=',', color='red')
                for j in range(1, len(chain.elements_id)):
                    id_node1 = chain[j-1]
                    id_node2 = chain[j]
                    x = [self.graph[id_node1].x, self.graph[id_node2].x]
                    y = [self.graph[id_node1].y, self.graph[id_node2].y]
                    plt.plot(x, y, marker=',', color='red')

        for id_terminal in self.graph.id_terminals:
            terminal = self.graph[id_terminal]
            plt.plot(terminal.x, terminal.y, marker='^', color='green')

        for id_distrib in self.graph.id_distribs:
            distrib = self.graph[id_distrib]
            plt.plot(distrib.x, distrib.y, marker='s', color='blue')
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.01)


if __name__ == '__main__':
    g = Graph()
    sol = Solution(g)
    print("cost : {}".format(sol.cost()))
    loop = sol.loops[0]
    #print(loop)
    sol.reverse(0, 2, 5)
    sol.heuristique()
    print(sol.isAdmissible())
    print("new cost", sol.cost())
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

    # sol.init_random_admissible()
    # print(sol.loops)
    # print(sol.chains)
    # print(sol.isAdmissible())
    # print(sol.cost())
    #
    # sol.write()

    # g = Graph()
    # sol = Solution(g)
    # print("cost : {}".format(sol.cost()))
    # sol.clusterize_distributions(5)
    # print(sol.id.id_distribs)
    # print(sol.id.id_terminals)
    # sol.write()
