from algo import *
import graph
from graph import Graph, Solution
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

    def test_disturb_between_chains(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_between_chains()
        self.assertTrue(sol.isAdmissible())

    def test_disturb(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb()
        self.assertTrue(sol.isAdmissible())


unittest.main()
