import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Function
import numpy as np


class GradientCalculator:
    def __init__(self):
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.model.eval()

    def preprocess_image(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(-1, 1, 1)
        return (x - mean) / std

    def get_gradient(self, x):

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = self.preprocess_image(x)
        x.requires_grad_()


        output = self.model(x)


        self.model.zero_grad()
        target_score = output[0, 292]
        target_score.backward()

        return np.array(x.grad).reshape(3,224,224)



if __name__ == "__main__":
    import numpy as np


    dummy_input = np.random.rand(3, 224, 224).astype(np.float32)

    target_class_index = 292

    grad_calc = GradientCalculator()
    gradient = grad_calc.get_gradient(dummy_input)

    print("Gradient shape:", gradient.shape)
