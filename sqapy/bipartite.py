import numpy as np


class BipartiteGraph:

    def __init__(self, b, c, W, initial_state=None):
        self.b = b
        self.c = c
        self.h = np.concatenate([b, c])

        if W.shape != (len(self.b), len(self.c)):
            raise ValueError("The shape of weights must be ({},{})".format(len(self.b), len(self.c)))
        else:
            self.W = W
        self.n_spins = len(self.h)
        self.spins = initial_state

        J1 = np.concatenate([np.zeros((self.W.shape[0], self.W.shape[0]), dtype=float), self.W.T], 0)
        J2 = np.concatenate([self.W, np.zeros((self.W.shape[1], self.W.shape[1]), dtype=float)], 0)
        self.J = np.concatenate([J1, J2], 1)
        self.linked = [[] for _ in range(self.n_spins)]
        for i, j in enumerate(self.J):
            for k, l in enumerate(j):
                if l != 0:
                    self.linked[i].append(k)
        if self.spins is not None:
            self.H = self.calculate_H()

    def initialize(self):
        if self.spins is None:
            self.spins = [int(np.random.random() > 0.5) for x in range(self.n_spins)]
            self.H = self.calculate_H()

    def calculate_H(self):
        H = 0
        for i in range(self.n_spins):
            Hi = self.h[i]
            Hi += 0.5 * sum((1 if self.spins[j] else -1) * self.J[i, j] for j in self.linked[i])
            if self.spins[i]:
                Hi *= -1
            H += Hi
        return H

    def calculate_dH(self, i):
        dH = self.h[i]
        dH += sum((1 if self.spins[j] else -1) * self.J[i, j] for j in self.linked[i])
        if not self.spins[i]:
            dH *= -1
        return 2 * dH

    def get_spin(self, i):
        return 1 if self.spins[i] else -1

    def flip_spin(self, i):
        self.spins[i] ^= True
