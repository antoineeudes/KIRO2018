from graph import Graph
from algo import SimulatedAnnealing, SimulatedAnnealing_log, SimulatedAnnealing_exp, Solution
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure, array, append, show
from config import PATH_EMPIRICAL_SEARCH


# class ParameterTestings:
#     def __init__(self, nb_iterT, nb_iterC, T_inf, T_sup, C_inf, C_sup):
#         self._nb_iterC = nb_iterC
#         self._nb_iterT = nb_iterT
#         self._T_inf = T_inf
#         self._T_sup = T_sup
#         self._C_inf = C_inf
#         self._C_sup = C_sup
#
#     def compute(self):
#         fichier = open(PATH_EMPIRICAL_SEARCH, 'w')
#         graph = Graph()
#         graph.get_reference()
#         first_solution = Solution(graph)
#         for T in np.linspace(self._T_inf, self._T_sup, self._nb_iterT):
#             for C in np.linspace(self._C_inf, self._C_sup, self._nb_iterC):
#                 Algo = SimulatedAnnealing_log(first_solution, T, C)
#                 cost = Algo.compute(show=False).cost()
#                 line = str(T) + '\t' + str(C) + '\t' + str(cost) + '\n'
#                 print(line)
#                 fichier.write(line)
#         fichier.close()
#
#     def plot3D(self):
#
#         """permet de determiner les meilleurs parametres alpha et pas"""
#
#         fig = figure()
#         ax = fig.gca(projection='3d')
#         Temperatures = array([])
#         coefs_C = array([])
#         Costs = array([])
#         fichier = open(PATH_EMPIRICAL_SEARCH, 'r')
#         for line in fichier:
#             M = []
#             J = line.strip().split('\t')
#             for x in J:
#                 M.append(float(x))
#             Temperatures = append(Temperatures, M[0])
#             coefs_C = append(coefs_C, M[1])
#             Costs = append(Costs, M[2])
#         fichier.close()
#         ax.scatter(Temperatures, coefs_C, zs=Costs, zdir='z', s=20.)
#         ax.set_xlabel('TEMPERATURE DE DEPART')
#         ax.set_ylabel('COEFFICIENT C')
#         ax.set_zlabel('COUTS')
#         show()


class ParameterTestings:
    def __init__(self, nb_iterT, nb_iteralpha, T_inf, T_sup, alpha_inf, alpha_sup):
        self._nb_iteralpha = nb_iteralpha
        self._nb_iterT = nb_iterT
        self._T_inf = T_inf
        self._T_sup = T_sup
        self._alpha_inf = alpha_inf
        self._alpha_sup = alpha_sup

    def compute(self):
        fichier = open(PATH_EMPIRICAL_SEARCH, 'w')
        graph = Graph()
        graph.get_reference()
        first_solution = Solution(graph)
        for T in np.linspace(self._T_inf, self._T_sup, self._nb_iterT):
            for alpha in np.linspace(self._alpha_inf, self._alpha_sup, self._nb_iteralpha):
                Algo = SimulatedAnnealing_exp(first_solution, T, alpha)
                cost = Algo.compute(show=False).cost()
                line = str(T) + '\t' + str(alpha) + '\t' + str(cost) + '\n'
                print(line)
                fichier.write(line)
        fichier.close()

    def plot3D(self):

        """permet de determiner les meilleurs parametres alpha et pas"""

        fig = figure()
        ax = fig.gca(projection='3d')
        Temperatures = array([])
        coefs_C = array([])
        Costs = array([])
        fichier = open(PATH_EMPIRICAL_SEARCH, 'r')
        for line in fichier:
            M = []
            J = line.strip().split('\t')
            for x in J:
                M.append(float(x))
            Temperatures = append(Temperatures, M[0])
            coefs_C = append(coefs_C, M[1])
            Costs = append(Costs, M[2])
        fichier.close()
        ax.scatter(Temperatures, coefs_C, zs=Costs, zdir='z', s=20.)
        ax.set_xlabel('TEMPERATURE DE DEPART')
        ax.set_ylabel('COEFFICIENT ALPHA')
        ax.set_zlabel('COUTS')
        show()

if __name__=='__main__':
    Test = ParameterTestings(50, 50, 0.01, 1, 0.7, 0.95)
    # Test.compute()
    Test.plot3D()


#
#
# g = Graph(100)
# costs = np.zeros(9)
# j = 0
# for alpha in np.linspace(0.7, 0.95, 3):
#
#     for T in np.linspace(100, 1000, 3):
#         print('alpha = '+ str(alpha))
#         print('T = ' + str(T))
#         S = SimulatedAnnealing(alpha, T, g)
#         min_solution = S.compute()
#
#         for i in range(1000):
#             solution = S.compute()
#             print(min_solution.cost())
#             if(solution.cost() < min_solution.cost()):
#                 min_solution = solution
#
#         costs[j] = min_solution.cost()
#         j += 1
#
# print(costs)
#
# x=np.unique(np.linspace(0.7, 0.95, 3))
# y=np.unique(np.linspace(100, 1000, 3))
# X,Y = np.meshgrid(x,y)
#
# Z=costs.reshape(len(y),len(x))
# print(Z)
#
# plt.pcolormesh(X,Y,Z)
#
# plt.show()
