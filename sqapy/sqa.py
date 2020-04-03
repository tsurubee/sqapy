import copy
import numpy as np


class SQASampler:

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

    @staticmethod
    def step_accepted(dH, T, dist=np.random.uniform):
        return (dH <= 0.0 or np.exp(-dH / T) > dist())
