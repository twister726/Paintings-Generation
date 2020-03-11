import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

class Encoder(nn.Module):
    
    def __init__(self, feature_dim):
        
        super(Encoder , self).__init__()
        
        resnet50 = tv.models.resnet50(pretrained = True)
        for child in resnet50.children():
            for param in child.parameters():
                param.requires_grad = False
            
            
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.avgpool = resnet50.avgpool
        self.fc = resnet50.fc
        num_ftrs = resnet50.fc.in_features

        self.fc = nn.Linear(num_ftrs, feature_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)      
        x = self.layer1(x)       
        x = self.layer2(x)       
        x = self.layer3(x)       
        x = self.layer4(x)       
        x = self.avgpool(x)       
        x = x.view(x.size(0),-1)
        y = self.fc(x)
        
        return y
