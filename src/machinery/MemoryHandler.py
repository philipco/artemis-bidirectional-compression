"""
Created by Philippenko, 17th September 2021.
"""
import numpy as np
import torch
from scipy.optimize import least_squares

from src.machinery.Parameters import Parameters
from src.machinery.TailAverager import AnytimeWindowsAverage3Acc, AnytimeExpoAverage


class MemoryHandler:

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.parameters = parameters
        self.expo_averager = AnytimeExpoAverage(parameters)
        self.awa_averager = AnytimeWindowsAverage3Acc(parameters)
        # self.coef_mem = [0 for i in range(len(self.h_i))]


    def optimal_memory(self, worker_mem, coef_mem):
        res = 0
        for i in range(len(coef_mem)):
            res += worker_mem[i] * coef_mem[i]
        return res

    def find_optimal_coef_mem(self, g_i, h_i):

        def memory(x):
            mem = 0
            for i in range(len(h_i)):
                mem += x[i] * h_i[i]
            return np.array(g_i - mem)

        self.coef_mem = [0 for i in range(len(h_i))]
        res_1 = least_squares(memory, self.coef_mem, bounds=(-1, 1))
        self.coef_mem = res_1.x

    def which_mem(self, h_i, averaged_h_i):
        # self.find_optimal_coef_mem()
        # if self.nb_it >= 0:
        #     self.delta_i = self.g_i - self.averaged_h_i
        # else:
        if not self.parameters.use_up_memory:
            return 0
        if self.parameters.use_averaged_h:
            return averaged_h_i
        else:
            return h_i
        # self.delta_i = self.g_i - self.averaged_h_i#self.optimal_memory(self.h_i, self.coef_mem)

    def update_average_mem(self, h_i, average_mem, nb_it):
        if self.parameters.use_unique_up_memory:
            new_val = h_i
        else:
            new_val = h_i[-1]
        if self.parameters.awa_tail_averaging:
            return self.awa_averager.compute_average(nb_it, new_val)
        elif self.parameters.expo_tail_averaging:
            return self.expo_averager.compute_average(nb_it, new_val)
        elif self.parameters.tail_averaging:
            assert not self.parameters.use_unique_up_memory, "When using true tail averaging, h_i should be the sequence of all h_i."
            n = (len(h_i) - 1) // 2
            return torch.mean(torch.stack(h_i[n:]), 0)
        else:
            rho = 0.95
            # Classic
            # return h_i
            # Weighted average
            # coef1 = rho * (1 - rho ** nb_it)  / (1 - rho ** (nb_it + 1))
            # coef2 = (1 - rho)  / (1 - rho ** (nb_it + 1))
            # return average_mem.mul(coef1) + h_i.mul(coef2)
            # Average
            return (1 - 1 / (nb_it + 1)) * average_mem + 1 / (nb_it + 1) * h_i


    def update_mem(self, h_i, averaged_h_i, quantized_delta):
        mem = h_i + self.parameters.up_learning_rate * quantized_delta
        if not self.parameters.debiased:
            return mem
        return mem + self.parameters.up_learning_rate * (averaged_h_i - h_i)