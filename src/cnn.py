import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            # nn.Dropout(.5), #0.2-0.5 for hidden and 0.2 max for input and conv
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # Never call on this. Instead, call on the variable object NeuralNetwork() since it does .__call__() and it uses this method
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        # pred_probab = nn.Softmax(dim=1)(logits)
        # y_pred = pred_probab.argmax(1)
        return logits