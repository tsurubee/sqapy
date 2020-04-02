import copy
import numpy as np

class Bipartite:

    def __init__(self, b, c, W, v_init=None):
        self.b = b
        self.c = c
        self.h = np.concatenate([b, c])

        if W.shape != (len(self.b), len(self.c)):
            raise ValueError("The shape of weights must be ({},{})".format(len(self.b), len(self.c)))
        else:
            self.W = W
        self.n_spins = len(self.h)
        self.spins = v_init

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
        return dH

    def get_spin(self, i):
        return 1 if self.spins[i] else -1

    def flip_spin(self, i):
        self.spins[i] ^= True


class BipartiteSampler:

    def __init__(self, model, P=40, T=0.015, G0=3, Gf=1e-6, e0=1e-6, ef=4, steps=1000):
        self.P  = P
        self.T  = T
        self.G0 = G0
        self.Gf = Gf
        self.e0 = e0
        self.ef = ef
        self.steps = steps
        self.model = model
        self.PT    = self.P * self.T

        schedule_G = np.linspace(G0, Gf, steps)
        schedule_e = np.linspace(e0, ef, steps)
        schedule_Jp = -0.5 * self.PT * np.log(np.tanh(schedule_G / (schedule_e * self.PT)))
        self.schedule = zip(schedule_Jp, schedule_G)

        self.slices = []
        for i in range(self.P):
            self.slices.append(copy.deepcopy(model))
            self.slices[i].initialize()

    def solve(self):
        for Jp, G in self.schedule:
            slices = np.random.permutation(self.P)
            for k in slices:
                i = np.random.randint(0, self.model.n_spins)
                dH_inner = self.slices[k].calculate_dH(i)
                dH_inter = self.calculate_dH_interslice(k, i, Jp)
                if self.step_accepted(dH_inner + dH_inter, self.PT):
                    self.slices[k].flip_spin(i)
            dH_global = np.zeros(self.P)
            i = np.random.randint(0, self.model.n_spins)
            for k in range(self.P):
                dH_global[k] = self.slices[k].calculate_dH(i)
            if self.step_accepted(np.sum(dH_global), self.PT):
                for k in range(self.P):
                    self.slices[k].flip_spin(i)

        energies = [s.calculate_H() for s in self.slices]
        best_index = np.argmin(energies)
        self.H = energies[best_index]
        self.spins = self.slices[best_index].spins.copy()

        return self.H, self.spins[:len(self.model.b)], self.spins[len(self.model.b):]

    def calculate_dH_interslice(self, k, i, Jp):
        dH = 2 * Jp * self.slices[k].get_spin(i)
        left = k - 1
        right = k + 1 if k + 1 < self.P else 0

        return dH * (self.slices[left].get_spin(i) + self.slices[right].get_spin(i))

    def step_accepted(self, dH, T, dist=np.random.uniform):
        return (dH <= 0.0 or np.exp(-dH / T) > dist())
