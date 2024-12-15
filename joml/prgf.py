from abc import ABC, abstractmethod

import numpy as np


class PRGF(ABC):
    def __init__(self, attack_model, surrogate_model, sigma, const_keep_factor=None, use_v=False):
        self._attack_model = attack_model
        self._surrogate_model = surrogate_model
        self._sigma = sigma
        self._D = None
        self._const_keep_factor = const_keep_factor
        self._use_v = use_v

    def __call__(self, x, y, q, *args, **kwargs):
        # main algorithm fun
        self._D = x.shape
        v = self._surrogate_model.get_gradient(x)
        if self._use_v:
            print("returning original (use_v set to True)...")
            return v

        if self._const_keep_factor:
            keep_factor = self._const_keep_factor
        else:
            alpha = self._cosine_similarity(x, y, v)
            keep_factor = self._get_lambda(alpha, q)

        if self._is_acceptable(keep_factor):
            print("returning original...")
            return v

        result = np.zeros(self._D)

        for i in range(q):
            result += self._calculate_shift(x, y, v, keep_factor)

        return self._final_correction(result, v, q, keep_factor)

    def _cosine_similarity(self, x, y, v):
        sigma = self._sigma
        above = (self._attack_model.f(np.clip(x + sigma*v, 0,1), y) - self._attack_model.f(x, y))/sigma
        D = self._D
        S = 100
        W = np.random.uniform(size=(S, *D))
        below = sum([((self._attack_model.f(np.clip(x + sigma*W[i,:,:,:], 0, 1), y) - self._attack_model.f(x, y))/sigma)**2 for i in range(S)])*(D[0]*D[1]*D[2])/S
        return above / np.sqrt(below)

    @abstractmethod
    def _get_lambda(self, alpha, q):
        pass

    @abstractmethod
    def _is_acceptable(self, keep_factor):
        pass

    @abstractmethod
    def _calculate_shift(self, x, y, v, keep_factor):
        pass

    @abstractmethod
    def _final_correction(self, result, v, q, keep_factor):
        pass
