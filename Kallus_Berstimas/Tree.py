'''
In this class we create a structure to hold the classification tree.
We need following requirements:
    0- An easy way for user to create the tree
    1- The set of branching nodes
    2- The set of terminal nodes
    3- For each node we need to get the index of left/right children and the parent

In this class we assume that we have a complete binary tree; we only receive the depth from the user
'''

import numpy as np
import math


class Tree:

    def __init__(self, d):
        self.depth = d
        self.Nodes = [i for i in range(1, np.power(2, d))]
        self.Terminals = [i for i in range(np.power(2, d), np.power(2, d + 1))]

    def get_left_children(self, n):
        if n in self.Nodes:
            return 2 * n
        else:
            raise Exception('Node index is not correct')

    def get_right_children(self, n):
        if n in self.Nodes:
            return 2 * n + 1
        else:
            raise Exception('Node index is not correct')

    def get_parent(self, n):
        if (n in self.Nodes) or (n in self.Terminals):
            return np.floor(n/2)
        else:
            raise Exception('Node index is not correct')


    """def get_ancestors(self,n):
        ancestors = []
        if (n in self.Nodes) or (n in self.Terminals):
            current = n
            while current != 1:
                current = int(np.floor(current/2))
                ancestors.append(current)
            return ancestors

        else:
            raise Exception('Node index is not correct')"""

    def get_ancestors(self, direction, n):
        current = n
        ancestors = []
        while current != 1:
            current_buffer = self.get_parent(current)
            if direction == 'r':
                if self.get_right_children(current_buffer) == current:
                    ancestors.append(current_buffer)
            else:
                if self.get_left_children(current_buffer) == current:
                    ancestors.append(current_buffer)
            current = current_buffer
        return ancestors


    def ancestors_dic(self):
        ancestors = {}
        for p in self.Terminals:
            ancestors[p] = [math.floor(p/(2**j)) for j in range(1, self.depth+1)]

        return ancestors

    def get_right_left(self):
        ancestor_rl = {}
        for i in self.Terminals:
            right = self.get_ancestors('r', i)
            for j in right:
                ancestor_rl[(i, j)] = 1
            left = self.get_ancestors('l', i)
            for j in left:
                ancestor_rl[(i, j)] = -1
        return ancestor_rl
