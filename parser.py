from graph import Graph
from algo import SimulatedAnnealing, SimulatedAnnealing_log, SimulatedAnnealing_exp, Solution
from config import PATH_NODE_FILE



def getNodes():
    fichier = open(PATH_NODE_FILE, 'r')
    L = []
    id = -1
    for line in fichier:
        J = [id] + line.strip().split(';')
        L.append(J)
        id += 1
    fichier.close()
    return L[1:]



if __name__=='__main__':
    print(getNodes())
