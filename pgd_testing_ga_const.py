import numpy as np
from PIL import Image
from joml.pgd import PGD
from surrogate_model import GradientCalculator
from attack_model import ImageClassifier

image_path = "tiger.jpg"
true_label = 292
image_np = np.array(Image.open(image_path))
image_np = np.array(Image.fromarray(image_np).resize((224, 224)))
image_np = image_np / 255.0
image_np = np.transpose(image_np, (2, 0, 1))
classifier = ImageClassifier()
grad_calc = GradientCalculator()

pgd = PGD(attack_model=classifier, surrogate_model=grad_calc, algorithm="PRGF-GA", sigma=0.1, const_keep_factor=0.5)

res = pgd.attack(image_np, label=true_label,stop_iter=1000, eps=0.9 , q=10, lr=0.001)
np.save("adv.npy", res[1])
print(res)