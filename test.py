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
        self.loop = Loop(elements_id, [])
        self.loop2 = Loop(elements_id2, [])
        self.loop.add_chain(Chain([13, 16, 18], self.loop, 7))
        self.loop.add_chain(Chain([5], self.loop, 3))
        self.loop.add_chain(Chain([], self.loop, 19))
        self.loop2.add_chain(Chain([1, 2, 6], self.loop, 56))

        # self.graph = graph.Graph()
        # self.sol = Solution(self.graph, [self.loop, self.loop2])

    def test_get_id_elements_with_chain(self):
        self.assertTrue(7 in self.loop.get_id_elements_with_chain())
        self.assertTrue(3 in self.loop.get_id_elements_with_chain())
        self.assertTrue(19 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 52 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 0 in self.loop.get_id_elements_with_chain())
        self.assertTrue(not 4 in self.loop.get_id_elements_with_chain())

    def test_add_chain(self):
        self.loop.add_chain(Chain([2], self.loop, 52))

        ids = self.loop.get_id_elements_with_chain()
        self.assertTrue(52 in ids)
        self.assertTrue(7 in ids)
        self.assertTrue(3 in ids)
        self.assertTrue(19 in ids)
        # self.assertEqual(len(self.loop.chains_dict[7]), 1)
        # self.assertEqual(len(self.loop.chains_dict[3]), 1)
        # self.assertEqual(len(self.loop.chains_dict[19]), 1)

    def test_remove_chain(self):
        chain = self.loop.loop_chains[0]
        previous_parent_id = chain.parent_node_id
        # print("previous {}".format(previous_parent_id))
        self.loop.remove_chain(chain)
        self.assertTrue(not previous_parent_id in self.loop.get_id_elements_with_chain())
        # self.assertEqual(chain.parent_loop, None)
        # self.assertEqual(chain.parent_node_id, None)

    def test_change_anchor_point(self):
        chain = self.loop.loop_chains[0]
        previous_parent_id = chain.parent_node_id
        # previous_nb_chains = len(self.loop.chains_dict[previous_parent_id])
        chain.change_anchor_node(52)

        self.assertTrue(chain.parent_node_id, 52)
        # self.assertTrue(not previous_parent_id in self.loop.chains_dict)
        ids = self.loop.get_id_elements_with_chain()
        self.assertTrue(52 in ids)

    def test_disturb_between_loops_example(self):
        loop1 = self.loop
        loop2 = self.loop2

        print("{} {}".format(len(loop1.elements_id), len(loop2.elements_id)))
        # print("type 1 : {}, type 2 : {}".format(type(self.graph[loop1[i]]), type(self.graph[loop2[j]])))
        i, j = 1, 2
        chains_1 = loop1.get_chains_with_parent_id(loop1[i])
        chains_2 = loop2.get_chains_with_parent_id(loop2[j])

        print("chaine 1 : {}".format(chains_1))
        print("chaine 2 : {}".format(chains_2))

        # Mise a jour des attributs des chaines :
        # Retrait des chaines des noeuds parents
        if chains_1 != None:
            for chain1 in chains_1:
                chain1.delete_from_parent_loop()
        if chains_2 != None:
            for chain2 in chains_2:
                chain2.delete_from_parent_loop()

        # Echanges des noeuds parents
        # new_solution.loops[idLoop1][i],  new_solution.loops[idLoop2][j] = new_solution.loops[idLoop2][j], new_solution.loops[idLoop1][i]
        loop1[i], loop2[j] = loop2[j], loop1[i]
        # On raccroche les chaines aux nouveaux noeuds parents
        if chains_1 != None:
            for chain1 in chains_1:
                chain1.parent_node_id = loop2[j]
                loop2.add_chain(chain1)

        if chains_2 != None:
            for chain2 in chains_2:
                chain2.parent_node_id = loop1[i]
                loop1.add_chain(chain2)

        self.assertEqual(loop1[i], 56)
        self.assertEqual(loop2[j], 7)
        self.assertEqual(chains_1[0].parent_node_id, 7)
        self.assertEqual(chains_2[0].parent_node_id, 56)
        self.assertEqual(chains_1[0].parent_loop, loop2)
        self.assertEqual(chains_2[0].parent_loop, loop1)


unittest.main()
