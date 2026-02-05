import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#From https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Flattens a contiguous range of dims into a tensor.(dim 1)
        # This initializes but doesnt pass in data bc forward() handles that
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(0.25)
        # )

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(9216, 128),  # 64*12*12 after conv+pool for 28x28 input
        #     nn.ReLU(),
        #     nn.Linear(128, 10)
        # )


    # Never call on this. Instead, call on the variable object NeuralNetwork() since it does .__call__() and it uses this method
    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)

        # pred_probab = nn.Softmax(dim=1)(logits)
        # y_pred = pred_probab.argmax(1)

        # x = self.conv_layers(x)
        # x = torch.flatten(x, 1)
        # model = self.fc_layers(x)


        return logits