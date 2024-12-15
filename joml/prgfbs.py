from functools import reduce

import numpy as np

from .prgf import PRGF

class PRGFBS(PRGF):
    def __init__(self, attack_model, surrogate_model, sigma, const_keep_factor=None, use_v=False):
        super().__init__(attack_model, surrogate_model, sigma, const_keep_factor, use_v)

    def _get_lambda(self, alpha, q):
        D = reduce(lambda x,y: x*y, self._D)
        al = 1/(D+2*q-2)
        ar = (2*q-1)/(D+2*q-2)
        if alpha**2 <= al:
            return 0
        if alpha**2 < ar:
            above = (1-alpha**2)*(alpha*alpha*(D+2*q-2)-1)
            below = 2*alpha*alpha*D*q-(alpha**4)*D*(D+2*q-2)-1
            return above/below
        return 1

    def _is_acceptable(self, keep_factor):
        return keep_factor == 1

    def _calculate_shift(self, x, y, v, keep_factor):
        D = self._D
        E = np.random.uniform(size=self._D)
        identity_matrices = np.repeat(np.eye(D[1]).reshape(1, D[1], D[2]), 3, axis=0)  # Shape (3, m, m)
        transposed_matrices = np.transpose(x, (0, 2, 1))  # Shape (3, n, m)
        product_matrices = np.matmul(x, transposed_matrices)
        product_matrices = np.matmul(product_matrices, E)
        result_matrices = np.sign(identity_matrices - product_matrices)
        u = np.sqrt(keep_factor) * v + np.sqrt(1 - keep_factor)*result_matrices

        # print(x.shape, u.shape)
        return u*(self._attack_model.f(np.clip(x + self._sigma*u,0,1), y) - self._attack_model.f(x, y))/self._sigma

    def _final_correction(self, result, v, q, keep_factor):
        return result/q
