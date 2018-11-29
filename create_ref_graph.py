from random import random
from graph import Graph
from copy import deepcopy
from config import PATH_REFERENCE_GRAPH


def create_random_graph(nb_vertex, width=1, height=1):
    G = Graph(width, height)
    G.randomize(nb_vertex)
    G.display()
    Vertices = []
    for i in range(nb_vertex):
        vertex = G[i]
        Vertices.append([vertex.x, vertex.y])
    return Vertices


if __name__=='__main__':
    fichier = open(PATH_REFERENCE_GRAPH, 'w')
    Vertices = create_random_graph(100)
    for x in Vertices:
        fichier.write(str(x[0]) + '\t' + str(x[1]) + '\n')
    fichier.close()
