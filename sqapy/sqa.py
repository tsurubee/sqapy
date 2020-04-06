import time
import copy
import numpy as np


class SQASampler:

    def __init__(self, model, trotter=40, temp=0.02, G0=3, Gf=1e-5, steps=1000):
        self.trotter  = trotter
        self.temp  = temp
        self.G0 = G0
        self.Gf = Gf
        self.steps = steps
        self.model = model

        self.schedule_G = np.linspace(G0, Gf, steps)
        self.schedule_Jp = 0.5 * np.log(np.tanh(self.schedule_G / (self.trotter * self.temp)))

        self.slices = []
        for i in range(self.trotter):
            self.slices.append(copy.deepcopy(model))
            self.slices[i].initialize()

    def sample(self, n_sample=1, reinitialize=True):
        energies = []
        spins = []
        self.execution_time = []
        for i in range(n_sample):
            start = time.time()
            if reinitialize and i != 0:
                for i in range(self.trotter):
                    self.slices[i].initialize()
            for Jp in self.schedule_Jp:
                slices = np.random.permutation(self.trotter)
                for k in slices:
                    i = np.random.randint(0, self.model.n_spins)
                    dH_inner = self.slices[k].calculate_dH(i)
                    dH_inter = self.calculate_dH_interslice(k, i, Jp)
                    if self.step_accepted(dH_inner + dH_inter, self.trotter * self.temp):
                        self.slices[k].flip_spin(i)
            end = time.time()
            self.execution_time.append((end - start) * 10**3)
            energy_list = [s.calculate_H() for s in self.slices]
            best_index = np.argmin(energy_list)
            energies.append(energy_list[best_index])
            spins.append(self.slices[best_index].spins)
        return energies, spins

    def calculate_dH_interslice(self, k, i, Jp):
        dH = -2 * self.slices[k].get_spin(i) * Jp
        left = k - 1
        right = k + 1 if k + 1 < self.trotter else 0

        return dH * (self.slices[left].get_spin(i) + self.slices[right].get_spin(i))

    @staticmethod
    def step_accepted(dH, T, dist=np.random.uniform):
        return (dH <= 0.0 or np.exp(-dH / T) > dist())
