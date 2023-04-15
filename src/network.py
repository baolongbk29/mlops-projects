
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VITBase(nn.Module):
    def __init__(self, model_name, n_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
    
class MobileNetBase(nn.Module):
        
    def __init__(self, model_name, n_class=2, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=n_features, out_features=n_class, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x