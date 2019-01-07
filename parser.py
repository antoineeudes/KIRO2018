from config import PATH_NODE_FILE, PATH_EDGE_FILE

def getNumberNodes():
    fichier = open(PATH_NODE_FILE, 'r')
    nbNodes = 0
    for line in fichier:
        nbNodes += 1
    fichier.close()
    return nbNodes-1


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


def getEdges():
    fichier = open(PATH_EDGE_FILE)
    distances = dict()
    nbNodes = getNumberNodes()
    file = fichier.readlines()
    for i in range(nbNodes):
        for j in range(nbNodes):
            distances[i, j] = int(file[j+nbNodes*i].strip())
    return distances



if __name__=='__main__':
    print(getEdges())
