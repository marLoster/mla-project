import numpy as np

from .prgf import PRGF

class PRGFGA(PRGF):
    def __init__(self, attack_model, surrogate_model):
        super().__init__(attack_model, surrogate_model)

    def _get_lambda(self, alpha, q, D):
        pass

    def _is_acceptable(self, keep_factor):
        pass

    def _calculate_shift(self, x, y, sigma, random_vector):
        pass

    def _final_correction(self, result, v, q, keep_factor):
        pass