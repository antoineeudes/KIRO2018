from algo import *
import graph
from graph import Graph, Solution
import unittest

def is_cycle(tab):
    seen = dict()
    for x in tab:
        seen[x]=True

    return len(seen) == len(tab)


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(100)
        self.sol = Solution(self.graph)

    def test_disturbs(self):
        sol = disturb(self.sol)
        sol2 = disturb2(self.sol)
        sol3 = disturb3(self.sol)
        sol4 = disturb4(self.sol)
        for i in range(10):
            sol = disturb(sol)
            sol2 = disturb2(sol2)
            sol3 = disturb3(sol3)
            sol4 = disturb3(sol4)
        self.assertTrue(is_cycle(sol._path_index))
        self.assertTrue(is_cycle(sol2._path_index))
        self.assertTrue(is_cycle(sol3._path_index))
        self.assertTrue(is_cycle(sol4._path_index))

    def test_cost_with_reverse(self):
        sol = copy.copy(self.sol)
        for i in range(10000):
            id1 = random.randint(0, sol.len-1)
            id2 = random.randint(0, sol.len-1)
            sol.reverse(id1, id2)
            diff = abs(sol.cost()-graph.real_cost(sol))
            print("{} {} {}".format(id1, id2, diff))
            self.assertTrue(diff<1e-5)


unittest.main()
