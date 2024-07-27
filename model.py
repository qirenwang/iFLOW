# import json
# import torch
# from paho.mqtt import client as mqtt_client
# from torch import nn, optim


# #Define model A
# class ModelA(nn.Module):
#     def __init__(self):
#         super(ModelA, self).__init__()
#         self.layer1 = nn.Linear(10, 20)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(20, 2)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         return x

# #Define model B
# class ModelB(nn.Module):
#     def __init__(self):
#         super(ModelB, self).__init__()
#         self.layer1 = nn.Linear(10, 15)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(15, 2)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         return x
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
