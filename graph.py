import random
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, ceil
from math import exp
import copy
from config import PATH_REFERENCE_GRAPH, PATH_REFERENCE_GRAPH_FIGURE, PATH_SOLUTION_FILE, PATH_START_SOLUTION_FILE, PATH_SAVE_SOLUTION_FILE
from parser import *
from shutil import copyfile

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

    for i in range(len(medians)):
        distance = distance2_euclidienne(medians[i][0], medians[i][1], vertex.x, vertex.y)
        if distance<distance_mini:
            distance_mini = distance
            id_cluster = i

    return id_cluster

def getClusterFromDistribs(distribs, vertex):
    id_cluster = 0
    distance_mini = float('inf')

    for i in range(len(distribs)):
        distrib = distribs[i]
        distance = distance2_euclidienne(distrib.x, distrib.y, vertex.x, vertex.y)
        if distance<distance_mini:
            distance_mini = distance
            id_cluster = i

    return id_cluster


def getClustersFromMedians(medians, graph):
    clusters = []
    for i in range(len(medians)):
        clusters.append([])

    for id, vertex in graph.vertex.items():
        clusters[getCluster(medians, vertex)].append(id)
    return clusters

def getClustersFromDistribs(distribs, graph):
    clusters = []
    for i in range(len(distribs)):
        clusters.append([])

    for id, vertex in graph.vertex.items():
        clusters[getClusterFromDistribs(distribs, vertex)].append(id)
    return clusters

class Loop: # Represente une unique boucle
    def __init__(self, graph = None, elements_id = [], loop_chains = []):
        self.graph = graph # Jamais modifie, passe par reference lors de la copie
        self.elements_id = elements_id # Contient les id des vertex composant la boucle
        self.loop_chains = loop_chains # Contient les chaines accrochees a la loop

    def __deepcopy__(self, memo):
        # new_loop = Loop(self.graph, self.elements_id[:], copy.deepcopy(self.loop_chains, memo))
        new_loop = Loop(self.graph, self.elements_id[:])
        memo['new_parent_loop_ref'] = new_loop
        new_loop_chains = copy.deepcopy(self.loop_chains, memo)
        new_loop.loop_chains = new_loop_chains
        return new_loop

    # def __copy__(self, memo):
    #     return Loop(self.graph, self.elements_id, copy.deepcopy(self.loop_chains))

    def create_chain(self, parent_node_id, chain_list):
        '''Créée une chaine a un element de la loop'''
        if parent_node_id in self.elements_id:
            chain = Chain(self.graph, chain_list, self, parent_node_id)
            chain.parent_loop = self
            self.loop_chains.append(chain)
            return chain
        else:
            print("set_chain : element pas dans la loop")

        return None

    def add_chain(self, chain):
        '''Ajoute un objet de type Chain a une Loop'''
        parent_node_id = chain.parent_node_id

        if parent_node_id in self.elements_id:
            chain.parent_loop = self
            self.loop_chains.append(chain)
        else:
            print("set_chain : element pas dans la loop")

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

    def cost_edge(self, id1, id2):
        return self.graph.edges[id1, id2]

    def cost_loop_only(self):
        '''Compute the cost of the loop without chain'''
        if len(self.elements_id) == 0:
            return 0

        cost = self.cost_edge(self[-1], self[0])

        for i in range(len(self.elements_id)-1):
            cost += self.cost_edge(self[i], self[i+1])

        return cost

    def cost(self):
        '''Compute the cost of the loop plus chains'''

        cost = self.cost_loop_only()

        for chain in self.loop_chains:
            cost += chain.cost()

        return cost


class Chain:
    def __init__(self, graph = None, elements_id = [], parent_loop = None, parent_node_id = None):
        self.graph = graph # Jamais modifie, passe par reference lors de la copie
        self.elements_id = elements_id
        self.parent_loop = parent_loop
        self.parent_node_id = parent_node_id

    def __deepcopy__(self, memo):
        if 'new_parent_loop_ref' in memo.keys():
            return Chain(self.graph, self.elements_id[:], memo['new_parent_loop_ref'], parent_node_id = self.parent_node_id)
        else:
            return Chain(self.graph, self.elements_id[:], copy.copy(self.parent_loop), parent_node_id = self.parent_node_id)

    # def __copy__(self):
    #     # On copie pas parent_loop car une copie d'une chaine fait reference a la meme parent_loop et pas a une copie de parent_loop
    #     return Chain(self.graph, self.elements_id[:], self.parent_loop, parent_node_id = self.parent_node_id)
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
        if self.parent_loop != None:
            self.parent_loop.remove_chain(self)
        self.parent_node_id = None
        self.parent_loop = None

    def cost_edge(self, id1, id2):
        return self.graph.edges[id1, id2]

    def cost(self):
        '''Compute the cost of the chain'''
        if len(self.elements_id) == 0:
            return 0

        cost = 0
        cost += self.cost_edge(self.parent_node_id, self[0])
        for i in range(len(self.elements_id)-1):
            cost += self.cost_edge(self[i], self[i+1])

        return cost



class Solution:
    def __init__(self, graph, loops=None):
        # La liste de vertex n'est jamais modifiée
        self.graph = graph
        self.loops = [] # Contient des objets de type Loop

        if loops != None:
            self.loops = loops

    # def __copy__(self):
    #     return Solution(self.graph, copy.copy(self.loops))

    def __deepcopy__(self, memo):
        return Solution(self.graph, copy.deepcopy(self.loops, memo))

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

        self.loops = []
        s = 0
        for cluster in clusters:
            s+=len(cluster)
            self.loops.append(Loop(self.graph, cluster, []))

        print(s==nbVertices, "GOALLLLLL")

        chains = []
        points_in_chains = []
        for i in range(nbClusters):
            n_i = len(clusters[i])
            if n_i > 30:
                points_in_chains = clusters[i][30:]
                nbChainsInLoop = ceil(len(points_in_chains)/4.)
                for k in range(nbChainsInLoop):
                    self.loops[i].create_chain(clusters[i][k], clusters[i][30+4*k:30+4*(k+1)])
                self.loops[i].create_chain(clusters[i][nbChainsInLoop], clusters[i][30+4*nbChainsInLoop:])
                for j in range(30, n_i):
                    del(clusters[i][-1])

        for i in range(nbClusters):
            nbDistrib = 0
            for j in range(len(clusters[i])):

                if self.graph.vertex[clusters[i][j]].isDistrib():
                    nbDistrib += 1
            print(nbDistrib)
        print(chains)

    def heuristique2(self):
        nbVertices = len(self.graph.vertex)
        nbClusters = len(self.graph.id_distribs)
        clusters = []

        distribs_list = [self.graph[id_distrib] for id_distrib in self.graph.id_distribs]
        clusters = getClustersFromDistribs(distribs_list, self.graph)

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
                clusters[i].insert(available_distribs[-1][-1], 0)
                del(clusters[available_distribs[-1][0]][available_distribs[-1][1]])
                del(available_distribs[-1])

        self.loops = []
        s = 0
        for cluster in clusters:
            s+=len(cluster)
            self.loops.append(Loop(self.graph, cluster, []))

        chains = []
        points_in_chains = []
        for i in range(nbClusters):
            n_i = len(clusters[i])
            if n_i > 30:
                points_in_chains = clusters[i][30:]
                nbChainsInLoop = ceil(len(points_in_chains)/4.)
                for k in range(nbChainsInLoop):
                    self.loops[i].create_chain(clusters[i][k], clusters[i][30+4*k:30+4*(k+1)])
                self.loops[i].create_chain(clusters[i][nbChainsInLoop], clusters[i][30+4*nbChainsInLoop:])
                for j in range(30, n_i):
                    del(clusters[i][-1])

        for i in range(nbClusters):
            nbDistrib = 0
            for j in range(len(clusters[i])):

                if self.graph.vertex[clusters[i][j]].isDistrib():
                    nbDistrib += 1

        print("Heuristique 2 done")

    # def cost_edge(self, id1, id2):
    #     return self.graph.edges[id1, id2]

    # def cost_loop(self, loop):
    #     '''Compute the cost of a given loop'''
    #
    #     if loop == []:
    #         return 0
    #
    #     cost = self.cost_edge(loop[-1], loop[0])
    #
    #     for i in range(len(loop.elements_id)-1):
    #         cost += self.cost_edge(loop[i], loop[i+1])
    #
    #     return cost

    # def cost_chain(self, chain):
    #     '''Compute the cost of a given chain'''
    #     if chain.elements_id == []:
    #         return 0
    #
    #     cost = 0
    #     cost += self.cost_edge(chain.parent_node_id, chain.elements_id[0])
    #     for i in range(len(chain.elements_id)-1):
    #         cost += self.cost_edge(chain.elements_id[i], chain.elements_id[i+1])
    #
    #     return cost

    def cost(self):
        cost = 0
        for loop in self.loops:
            cost += loop.cost()
            # for chain in loop.loop_chains:
            #     cost += chain.cost()

        return cost

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
        new_solution = copy.deepcopy(self)
        idLoop = self.getRandomIdLoop()
        i = random.randint(0, len(new_solution.loops[idLoop].elements_id)-1)
        j = random.randint(0, len(new_solution.loops[idLoop].elements_id)-1)
        new_solution = new_solution.reverse(idLoop, i, j) #Aucune influence sur les chaines
        # if not new_solution.is_loop_admissible(new_solution.loops[idLoop]):
        #     return self
        # else:
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

            new_solution = copy.deepcopy(self)
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
        if len(chain.elements_id)<=1:
            return self
        new_solution = copy.deepcopy(self)

        i = random.randint(0, len(chain.elements_id)-1)
        j = random.randint(0, len(chain.elements_id)-1)
        chain[i],  chain[j] = chain[j], chain[i]

        return new_solution

    def disturb_remove_from_chain_to_loop(self):
        '''Essai d'enelever un element d'une chaine pour le mettre dans la boucle'''
        idLoop = self.getRandomIdLoop()
        if len(self.loops[idLoop].elements_id) > 30:
            return self #Plus de place

        if len(self.loops[idLoop].loop_chains) == 0:
            return self # Aucune chaine dans la loop

        new_solution = copy.deepcopy(self)
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

    # def disturb_remove_from_chain_to_another_loop(self):
    #     idDestinationLoop = self.getRandomIdLoop()
    #     if len(self.loops[idDestinationLoop].elements_id) > 30:
    #         return self #Plus de place
    #     idOriginLoop = self.getRandomIdLoop()
    #     if len(self.loops[idOriginLoop].loop_chains) == 0:
    #         return self # Aucune chaine dans la loop
    #
    #     new_solution = copy.deepcopy(self)
    #     destinationLoop = new_solution.loops[idDestinationLoop]
    #     originLoop = new_solution.loops[idOriginLoop]
    #     i_chain = random.randint(0, len(originLoop.loop_chains)-1)
    #     chain = originLoop.loop_chains[i_chain]
    #     if len(chain.elements_id) == 0:
    #         return self # chaine choisie vide
    #     idElementInChain = random.randint(0, len(chain.elements_id))
    #     elementInChain = chain.elements_id[idElementInChain]
    #

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
        new_solution = copy.deepcopy(self)
        # new_solution = self
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
            # chain = Chain(self.graph, [element_id], loop, new_parent_id)
            del(loop.elements_id[i])
            # loop.loop_chains.append(chain)
            # chain = new_solution.loops[idLoop].create_chain(new_parent_id, [element_id])
            chain = loop.create_chain(new_parent_id, [element_id])
            # del(chain.parent_loop.elements_id[i])
            # new_solution.loops[idLoop][0] = 0
            # print("{} {}".format(chain.parent_node_id, chain.parent_loop.elements_id))
            # print("{} {}".format(chain.parent_node_id, loop.elements_id))
            # if not chain.parent_node_id in chain.parent_loop.elements_id:
            #     print("erreur")
            #     quit()
            return new_solution

        elif len(chains_by_id[element_id]) == 1: # Une unique chaine partant de l'element selectionne
            chain = chains_by_id[element_id][0]
            if len(chain.elements_id) >= 5: # Plus de place
                return self

            chain.elements_id.insert(0, element_id) # Ajout de l'element a la boucle au debut de celle-ci
            chain.parent_node_id = new_parent_id
            del(loop.elements_id[i])


            return new_solution

        return self

    def disturb(self):
        i = random.randint(0, 8)

        if i == 0:
            return self.disturb_transfer_from_chain_to_chain()
        elif i == 1:
            return self.disturb_remove_from_chain_to_loop()
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
            return self.disturb_between_chains()
        elif i == 8:
            return self.disturb_in_loop()
        return self

    def disturb_between_chains(self):
        new_solution = copy.deepcopy(self)

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
        if len(self.loops[id_loop].loop_chains[id_chain1].elements_id) == 0:
            return self

        p = random.randint(0, 1)
        if p == 0: # Creation d'une nouvelle chaine a partir de cet element
            new_solution = copy.deepcopy(self)
            id_loop2 = self.getRandomIdLoop()
            loop2 = new_solution.loops[id_loop2]
            chain1 = new_solution.loops[id_loop].loop_chains[id_chain1]
            i1 = random.randint(0, len(chain1.elements_id)-1)

            pos_anchor_point = random.randint(0, len(loop2.elements_id)-1)
            loop2.create_chain(loop2.elements_id[pos_anchor_point], [chain1.elements_id[i1]])
            del(chain1.elements_id[i1])

            return new_solution

        elif p == 1:
            id_chain2 = random.randint(0, len(self.loops[id_loop].loop_chains)-1)

            if id_chain1 == id_chain2:
                return self

            if len(self.loops[id_loop].loop_chains[id_chain2].elements_id) >= 5:
                return self # Plus de place

            new_solution = copy.deepcopy(self)

            chain1 = new_solution.loops[id_loop].loop_chains[id_chain1]
            chain2 = new_solution.loops[id_loop].loop_chains[id_chain2]

            i1 = random.randint(0, len(chain1.elements_id)-1)
            pos2 = random.randint(0, len(chain2.elements_id))

            chain2.elements_id.insert(pos2, chain1.elements_id[i1])
            del(chain1.elements_id[i1])

            return new_solution

        return self


    def disturb_anchor_point_in_loop(self):
        new_solution = copy.deepcopy(self)
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
        new_solution = copy.deepcopy(self)
        chain = new_solution.getRandomChain()
        if chain == None:
            return self
        new_loop = new_solution.loops[new_solution.getRandomIdLoop()]
        chain.delete_from_parent_loop()
        new_parent_id = random.randint(0, len(new_loop.elements_id)-1)
        chain.parent_node_id = new_loop.elements_id[new_parent_id]
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

        for chain in loop.loop_chains:
            if not chain.parent_node_id in loop.elements_id:
                print("Parent id {} pas dans la loop {}".format(chain.parent_node_id, loop.elements_id))
                return False

        return True

    def is_chain_admissible(self, chain):
        if not chain.parent_node_id in chain.parent_loop.elements_id:
            print("Chaine non admissible car parent_node_id {} pas dans parent_loop : {}".format(chain.parent_node_id, chain.parent_loop.elements_id))
            return False

        if chain.parent_node_id in chain.elements_id:
            print("Parent dans la chaine")
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
            chains[i][0] = id_loop
            chains[i][1].append(self.loops[id_loop][id_vertex])

        k = 0
        id_chain = 0
        while nb_terminals_added < nb_terminals:
                if k >= 5:
                    k=0
                    id_chain += 1
                chains[id_chain][1].append(self.graph.id_terminals[nb_terminals_added])
                k += 1
                nb_terminals_added += 1

        self.chains = chains

    def write(self, init_overwrite = False, save = False):
        print("Wirtting...")
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

            for chain in loop.loop_chains:
                if chain.elements_id == []:
                    continue
                line = "c" + " " + str(chain.parent_node_id)
                for id in chain.elements_id:
                    line += " " + str(id)
                line += "\n"
                fichier.write(line)

        fichier.close()

        if init_overwrite:
            copyfile(PATH_SOLUTION_FILE, PATH_START_SOLUTION_FILE)

        if save:
            copyfile(PATH_SOLUTION_FILE, PATH_SAVE_SOLUTION_FILE+str(self.cost())+".txt")

        print("Writing done")

    def read(self):
        print("Reading...")

        try:
            fichier = open(PATH_START_SOLUTION_FILE, 'r')
        except OSError as err:
            print("Reading failed : {}".format(err))
            return False # Fail to open

        for line in fichier:
            data = line.strip().split(' ')
            if data[0] == 'b': #Boucle
                data_int = [int(id) for id in data[1:]]
                loop = Loop(self.graph, data_int, [])
                self.loops.append(loop)

        fichier.seek(0) #Retour au debut
        for line in fichier:
            data = line.strip().split(' ')
            if data[0] == 'c': #Chaine
                # Recherche de la boucle a laquelle est ratachee la chaine
                parent_node_id = int(data[1])
                for i in range(len(self.loops)):
                    if parent_node_id in self.loops[i].elements_id:
                        data_int = [int(id) for id in data[2:]]
                        self.loops[i].create_chain(parent_node_id, data_int)
        print("Reading done")
        return True

    def show(self, block=True):
        print("Drawing...")
        plt.clf()

        for i in range(len(self.loops)):
            loop = self.loops[i]
            for j in range(len(loop.elements_id)):
                id_node1 = loop[j-1]
                id_node2 = loop[j]
                x = [self.graph[id_node1].x, self.graph[id_node2].x]
                y = [self.graph[id_node1].y, self.graph[id_node2].y]
                plt.plot(x, y, marker=",", color='black')#colors[i%nb_colors])

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
            plt.annotate(str(id_terminal), (terminal.x, terminal.y))

        for id_distrib in self.graph.id_distribs:
            distrib = self.graph[id_distrib]
            plt.plot(distrib.x, distrib.y, marker='s', color='blue')
            plt.annotate(str(id_distrib), (distrib.x, distrib.y))

        print("Drawing done")

        if block:
            plt.show(block=True)
        else:
            plt.pause(0.01)

if __name__ == '__main__':
    g = Graph()
    sol = Solution(g)
    sol.heuristique2()
    print("cost : {}".format(sol.cost()))
    loop = sol.loops[0]
    sol.reverse(0, 2, 5)
    print(sol.isAdmissible())
    print("new cost", sol.cost())
