from functools import reduce

import numpy as np

from .prgf import PRGF

class PRGFGA(PRGF):
    def __init__(self, attack_model, surrogate_model, sigma, const_keep_factor=None, use_v=False):
        super().__init__(attack_model, surrogate_model, sigma, const_keep_factor, use_v)

    def _get_lambda(self, alpha, q):
        D = reduce(lambda x, y: x * y, self._D)
        Eb = np.sqrt(q/(D+q-1))
        upper = alpha*(1-Eb*Eb)
        lower = alpha*(1-Eb*Eb) + (1-alpha*alpha)*Eb
        return upper/lower

    def _is_acceptable(self, keep_factor):
        c = 0.6
        return keep_factor >= c

    def _calculate_shift(self, x, y, sigma, random_vector):
        u = np.random.uniform(size=self._D)
        return u * (self._attack_model.f(np.clip(x + self._sigma * u, 0, 1), y) - self._attack_model.f(x,
                                                                                                       y)) / self._sigma

    def _final_correction(self, result, v, q, keep_factor):
        return keep_factor*v + (1-keep_factor)*result