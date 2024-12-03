from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch import nn
from PIL import Image
import numpy as np


class ImageClassifier:
    def __init__(self, model_name="microsoft/resnet-50"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)
        self.criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss object

    def pred(self, x):
        inputs = self.processor(x, return_tensors="pt", do_rescale=False)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return logits

    def pred_class(self, x):
        inputs = self.processor(x, return_tensors="pt", do_rescale=False)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.argmax(logits, dim=-1).item()

    def f(self, x, y):
        logits = self.pred(x)
        loss = self.criterion(logits, torch.tensor([y]))
        return loss.item()


if __name__ == "__main__":
    image_path = "tiger.jpg"
    true_label = 292


    image_np = np.array(Image.open(image_path))
    print(image_np.shape)

    classifier = ImageClassifier()

    logits = classifier.pred(image_np)

    print(logits)
    predicted_label = logits.argmax(-1).item()
    print(logits[0,predicted_label])
    print(predicted_label)
    print(classifier.model.config.id2label[predicted_label])

    loss = classifier.f(image_np, true_label)
    print(f"Cross-entropy loss: {loss.item()}")
