import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import pytorch

from torchvision.transforms import transforms
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import math
import time

from src.tutorial_nn import NeuralNetwork
from src.tutorial_run import *


# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
NUM_EPOCHS = 1
ALPHA_LEARNING = 1 * 10^-3
BATCH_SIZE = 4

# --------------------------------------------------------------------------- #
# MAIN WALK
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nn.ConvTranspose3d()

def main():
    if torch.cuda.is_available():
        print("CUDA device found")
        device_string = "cuda"
    else:
        print("No CUDA found. Resorting to CPU")
        device_string = "cpu"

    device = torch.device(device_string)
    
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    model = NeuralNetwork().to(device)
    
    print(model)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA_LEARNING)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, BATCH_SIZE)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == '__main__':
    main()