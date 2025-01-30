import torch
import torch.nn as nn

class TextLinearModel(nn.Module):
    def __init__(self):
        super(TextLinearModel, self).__init__()
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_features):
        x = self.linear(text_features)
        x = self.sigmoid(x)
        return x
    
# class TBMEncoder(nn.Module):
#     def __init__(self):
#         super(TBMEncoder, self).__init__()
#         self.classifiers = nn.ModuleList([nn.Sequential(
#             nn.Linear(1408, 1),
#             nn.Sigmoid()
#         ) for _ in range(300)])

#     def forward(self, image_features, classifier_index):
#         return self.classifiers[classifier_index](image_features)

class TBMPredictor(nn.Module):
    def __init__(self):
        super(TBMPredictor, self).__init__()
        self.linear = nn.Linear(18, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_per_text):
        x = self.linear(logits_per_text)
        x = self.sigmoid(x)
        return x
    
class MiRAGeTxt(nn.Module):
    def __init__(self):
        super(MiRAGeTxt, self).__init__()
        self.linear = nn.Linear(19, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_per_text):
        x = self.linear(logits_per_text)
        x = self.sigmoid(x)
        return x