from algo import *
import graph
from graph import Graph, Solution, Loop, Chain
import unittest
import copy

def is_cycle(tab):
    seen = dict()
    for x in tab:
        seen[x]=True

    return len(seen) == len(tab)


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.graph = graph.Graph()
        self.sol = Solution(self.graph)
        self.sol.heuristique()

    def test_disturb_in_loop(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_in_loop()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_between_loops(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_between_loops()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_in_chain(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_in_chain()
        self.assertTrue(sol.isAdmissible())

    # def test_disturb_between_chains(self):
    #     sol = copy.copy(self.sol)
    #     for i in range(1000):
    #         sol = sol.disturb_between_chains()
    #     self.assertTrue(sol.isAdmissible())
    #
    # def test_disturb(self):
    #     sol = copy.copy(self.sol)
    #     for i in range(1000):
    #         sol = sol.disturb()
    #     self.assertTrue(sol.isAdmissible())

class TestLoop(unittest.TestCase):
    def setUp(self):
        elements_id = [3, 7, 4, 0, 19, 52]
        chains_dict = dict()

        self.loop = Loop(elements_id)
        self.loop.add_chain(Chain([13, 16, 18], self.loop, 7, self.loop.all_chains))
        self.loop.add_chain(Chain([5], self.loop, 3, self.loop.all_chains))
        self.loop.add_chain(Chain([], self.loop, 19, self.loop.all_chains))

    def test_get_id_elements_with_chain(self):
        print(self.loop.chains_dict)
        self.assertTrue(7 in self.loop.get_id_elements_with_chain())
        self.assertTrue(3 in self.loop.get_id_elements_with_chain())
        self.assertTrue(19 in self.loop.get_id_elements_with_chain())

    def test_get_list_of_chains(self):
        print(self.loop.chains_dict)
        print("test")
        print(self.loop.get_list_of_chains())
        self.assertTrue([5] in self.loop.get_list_of_chains())

unittest.main()
