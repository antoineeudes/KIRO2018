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
        self.sol.heuristique2()

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

    def test_disturb_transfer_from_chain_to_chain(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_transfer_from_chain_to_chain()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_anchor_point_in_loop(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_anchor_point_in_loop()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_remove_from_chain_to_loop(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_remove_from_chain_to_loop()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_create_new_chain(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_create_new_chain()
        self.assertTrue(sol.isAdmissible())

    def test_disturb_anchor_point_in_other_loop(self):
        sol = copy.copy(self.sol)
        for i in range(1000):
            sol = sol.disturb_anchor_point_in_other_loop()
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
        elements_id2 = [23, 26, 56, 65, 89, 32]
        chains_dict = dict()
        self.loop = Loop(None, elements_id, [])
        self.loop2 = Loop(None, elements_id2, [])
        self.loop.add_chain(Chain(None, [13, 16, 18], self.loop, 7))
        self.loop.add_chain(Chain(None, [5], self.loop, 3))
        self.loop.add_chain(Chain(None, [], self.loop, 19))
        self.loop2.add_chain(Chain(None, [1, 2, 6], self.loop, 56))

    def test_get_id_elements_with_chain(self):
        self.assertTrue(7 in self.loop.get_id_elements_with_chain())
        self.assertTrue(3 in self.loop.get_id_elements_with_chain())
        self.assertTrue(19 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 52 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 0 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 4 in self.loop.get_id_elements_with_chain())

    def test_add_chain(self):
        self.loop.add_chain(Chain(None, [2], self.loop, 52))

        ids = self.loop.get_id_elements_with_chain()
        self.assertTrue(52 in ids)
        self.assertTrue(7 in ids)
        self.assertTrue(3 in ids)
        self.assertTrue(19 in ids)

    def test_remove_chain(self):
        chain = self.loop.loop_chains[0]
        previous_parent_id = chain.parent_node_id
        self.loop.remove_chain(chain)
        self.assertTrue(not previous_parent_id in self.loop.get_id_elements_with_chain())

    def test_change_anchor_point(self):
        chain = self.loop.loop_chains[0]
        previous_parent_id = chain.parent_node_id
        chain.change_anchor_node(52)

        self.assertTrue(chain.parent_node_id, 52)
        ids = self.loop.get_id_elements_with_chain()
        self.assertTrue(52 in ids)


unittest.main()
