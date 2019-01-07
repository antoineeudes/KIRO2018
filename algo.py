import graph
import random
from graph import Solution
from math import exp
from math import log
from matplotlib import pyplot as plt
import copy
import time

def indice_min(liste):
    mini = float('inf')
    n = len(liste)
    indice = 0
    for i in range(n):
        if liste[i]<mini:
            mini = liste[i]
            indice = i

    return indice, mini


def disturb_reverse(sol):
    s2 = copy.copy(sol)
    id1 = random.randint(0, sol.len-1)
    id2 = random.randint(0, sol.len-1)
    s2.reverse(id1, id2)
    return s2


class SimulatedAnnealing:
    def __init__(self, s0, T):
        self.T = T
        self.min_solution = s0
        self.start_time = None

    def reduce_temperature(self, T):
        return T

    def stopping_condition(self):
        # return self.T <= 10
        return False

    def timeout(self):
        if time.time()-self.start_time > 10:
            print("\n Stopped because timeout \n")
            return True
        return False

    def compute(self, start_solution=None, show=True, display_improvment=False):

        if(start_solution != None):
            self.min_solution = start_solution

        current_solution = copy.deepcopy(self.min_solution)

        self.start_time = time.time()

        while not self.stopping_condition() and not self.timeout():
            new_solution = current_solution.disturb()
            p = random.random()

            if p < exp(-max(0,new_solution.cost()-current_solution.cost())/self.T):
                current_solution = new_solution
                if current_solution.cost() < self.min_solution.cost():
                    self.min_solution = copy.deepcopy(current_solution)
                    if display_improvment:
                        current_solution.show(block=False)

            if(show):
                print("{} {} {}".format(self.min_solution.cost(), current_solution.cost(), self.T))
            self.T = self.reduce_temperature(self.T)

        return self.min_solution

class SimulatedAnnealing_exp(SimulatedAnnealing):
    def __init__(self, s0, T=100, alpha=0.9):
        super().__init__(s0, T)
        self.alpha = alpha
        self.previous_solution = self.min_solution
        self.nb_stab_iterations = 0

    def reduce_temperature(self, T):
        return self.alpha*T

    def stopping_condition(self):
        return self.T<1

    # def stopping_condition(self):
    #     if self.previous_solution == self.min_solution:
    #         self.nb_stab_iterations += 1
    #         # print("stable {}".format(self.nb_stab_iterations))
    #     else:
    #         self.nb_stab_iterations = 0
    #     self.previous_solution = self.min_solution
    #
    #     if self.nb_stab_iterations >= 10000 or self.T == 0:
    #         self.nb_stab_iterations = 0
    #         print("\n Stopped because stable \n")
    #         return True
    #     return False

class SimulatedAnnealing_log(SimulatedAnnealing):
    def __init__(self, s0, T0=100, C=None):
        self.T0 = T0 #Température initiale
        super().__init__(s0, T0)
        self.i = 1
        self.previous_solution = self.min_solution
        self.nb_stab_iterations = 0
        self.C = self.T0
        if C!=None:
            self.C = C


    def ResetTemperature(self, T0):
        self.i = 1
        self.T0 = T0

    def reduce_temperature(self, T):
        self.i += 1
        return self.C/log(self.i)

    # def stopping_condition(self):
    #     if self.previous_solution == self.min_solution:
    #         self.nb_stab_iterations += 1
    #     else:
    #         self.nb_stab_iterations = 0
    #
    #     self.previous_solution = self.min_solution
    #
    #     if self.nb_stab_iterations >= 500000:
    #         print("\n Stopped because stable \n")
    #         self.nb_stab_iterations = 0
    #         return True
    #     return False


class SimulatedAnnealing_repeated(SimulatedAnnealing_exp):
    def __init__(self, s0, T, alpha, nb):
        self.nb_annealing = nb
        self.T0 = T
        super().__init__(s0, T, alpha)

    # def stopping_condition(self):
    #     return self.T < 1

    def timeout(self):
        if time.time()-self.start_time > 20:
            print("\n Stopped because timeout \n")
            return True
        return False

    def compute(self, show=True, display_improvment=False):
        min_solution = copy.copy(self.min_solution)
        for i in range(self.nb_annealing):
            solution = super().compute(min_solution, show=False)
            self.T = self.T0 # On repart de la température initiale (pas fait automatiquement)
            if(solution.cost() < min_solution.cost()):
                min_solution = copy.deepcopy(solution)
            if(show):
                print("{} {}".format(min_solution.cost(), i))

        return min_solution


if __name__ == '__main__':
    g = graph.Graph()
    min_solution = Solution(g)
    # min_solution.show()
    if not min_solution.read(): #Essaie de lire une eventuelle solution de depart
        min_solution.heuristique2() #Si non trouve la construit par l'heuristique
    print(min_solution.cost())
    print("Is admissible : ", min_solution.isAdmissible())

    # S = SimulatedAnnealing_exp(min_solution)
    # S = SimulatedAnnealing_exp(min_solution)

    # S = SimulatedAnnealing_exp(min_solution, T=1000, alpha=0.9)
    # S = SimulatedAnnealing_repeated(min_solution, 10000, 0.99, 50)
    S = SimulatedAnnealing_log(min_solution, T0=100)

    # S = SimulatedAnnealing_repeated(min_solution, 1000, 0.3, 5000)

    # min_solution.show()
    time0 = time.time()
    min_solution = S.compute(display_improvment=False)
    min_solution.write(init_overwrite = True, save=True)
    S.ResetTemperature(100)
    min_solution = S.compute(display_improvment=False)
    min_solution.write(init_overwrite = True, save=True)
    S.ResetTemperature(100)
    min_solution = S.compute(display_improvment=False)
    min_solution.write(init_overwrite = True, save=True)
    S.ResetTemperature(100)
    min_solution = S.compute(display_improvment=False)
    min_solution.write(init_overwrite = True, save=True)
    S.ResetTemperature(100)
    min_solution = S.compute(display_improvment=False)

    # min_solution.show()
    min_solution.write(init_overwrite = True, save=True)

    print("Is admissible : {}".format(min_solution.isAdmissible()))
    print("Temps : {}".format(time.time()-time0))
    print("Cost : {}".format(min_solution.cost()))

    # for loop in min_solution.loops:
    #     print("loop {}".format(loop.elements_id))
    #     for chain in loop.loop_chains:
    #         print("{} : {} ".format(chain.parent_node_id, chain.parent_loop.elements_id))
    #         print(chain.parent_loop == loop)

    # min_solution.show()
