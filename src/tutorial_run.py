from typing import Callable

import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, Tensor
from torch.optim import Optimizer

from .tutorial_nn import NeuralNetwork

def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    # batch_size: int
):    
    size = len(dataloader)
    device = next(model.parameters()).device

    total_loss = 0
    all_preds = []
    all_labels = []

    # set model to training mode
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop (never will i code it again. ty DAG & PyTorch )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        total_loss += loss.item()
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / size
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def test_loop(
    dataloader: DataLoader, 
    model: nn.Module, 
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    labels_map = None
):
    # set model to evaluation mode (look into batch_normalization)
    model.eval()
    size = len(dataloader)

    total_loss = 0
    all_preds = []
    all_labels = []

    device = next(model.parameters()).device


    # serves to reduce unnecessary gradient computations and memory usage for tensors using requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / size
    accuracy = accuracy_score(all_labels, all_preds)

    report = {}

    if labels_map:
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        report = classification_report(y_true, y_pred, target_names=labels_map.values(), output_dict=True)
        # print("\nDetailed Classification Report:")
        # print(report)
        # return avg_loss, accuracy, all_preds, all_labels, report

    return avg_loss, accuracy, all_preds, all_labels, report