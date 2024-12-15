import numpy as np

from joml.prgfbs import PRGFBS
from joml.prgfga import PRGFGA


class PGD:
    def __init__(self, attack_model, surrogate_model, algorithm, **algorithm_keywords):
        self._attack_model = attack_model
        self._surrogate_model = surrogate_model
        self._algorithm = algorithm
        self._algorithm_keywords = algorithm_keywords

        # initialize prgf
        if algorithm == "PRGF-BS":
            self._prgf = PRGFBS(attack_model, surrogate_model, **algorithm_keywords)
        elif algorithm == "PRGF-GA":
            self._prgf = PRGFGA(attack_model, surrogate_model, **algorithm_keywords)
        else:
            raise TypeError(f"Invalid algorithm name {algorithm}")

    def attack(self, image, label, stop_iter=1000, eps=0.1, q=10, lr=0.1):
        prev_image = image
        adversarial_image = None
        for i in range(stop_iter):
            if not i%10:
                print(f"iter: {i}")
            estimated_gradient = self._prgf(prev_image, label, q)
            if np.sum(estimated_gradient.reshape(-1)) == 0:
                print("estimated_gradient = 0")
            adversarial_image = prev_image + lr*estimated_gradient
            adversarial_image = np.clip(adversarial_image, image*(1-eps), image*(1+eps))
            adversarial_image = np.clip(adversarial_image, 0, 1)
            res = self._attack_model.pred_class(adversarial_image)
            if res != label:
                print("succes")
                return True, adversarial_image
            else:
                prev_image = adversarial_image
                print(f"loss: {self._attack_model.f(adversarial_image, label)}")

        print("model was not broken")
        return False, adversarial_image