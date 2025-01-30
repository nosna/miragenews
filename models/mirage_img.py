import torch
import torch.nn as nn

class ImageLinearModel(nn.Module):
    def __init__(self):
        super(ImageLinearModel, self).__init__()
        self.linear = nn.Linear(1408, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features):
        x = self.linear(image_features)
        x = self.sigmoid(x)
        return x
    
class ObjectClassCBMEncoder(nn.Module):
    def __init__(self):
        super(ObjectClassCBMEncoder, self).__init__()
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(1408, 1),
            nn.Sigmoid()
        ) for _ in range(300)])

    def forward(self, image_features, classifier_index):
        return self.classifiers[classifier_index](image_features)

class ObjectClassCBMPredictor(nn.Module):
    def __init__(self):
        super(ObjectClassCBMPredictor, self).__init__()
        self.linear = nn.Linear(300, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_per_image):
        x = self.linear(logits_per_image)
        x = self.sigmoid(x)
        return x
    
class MiRAGeImg(nn.Module):
    def __init__(self):
        super(MiRAGeImg, self).__init__()
        self.linear = nn.Linear(301, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_per_image):
        x = self.linear(logits_per_image)
        x = self.sigmoid(x)
        return x