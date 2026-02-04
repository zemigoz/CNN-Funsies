import math
import time
import os
import gc

from collections import Counter
from pathlib import Path

import torch
import torchvision
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor

from src.tutorial_nn import NeuralNetwork
from src.tutorial_run import *
from src.analysis import *


# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

WANDB_API_KEY_PATH = BASE_DIR / Path("wandb_api_key.txt")

RNG_SEED = 314

FOLDS = 5

EPOCHS = 5
BATCH_SIZE = 64
ALPHA = 1e-4
# --------------------------------------------------------------------------- #
# MAIN WALK
# --------------------------------------------------------------------------- #
torch.manual_seed(RNG_SEED)

def main():
    if torch.cuda.is_available():
        print("CUDA device found")
        device_string = "cuda"
    else:
        print("No CUDA found. Resorting to CPU")
        device_string = "cpu"
    device = torch.device(device_string)


    os.environ["WANDB_API_KEY"] = WANDB_API_KEY_PATH.read_text().strip()
    wandb_group_name = Path(f"MNIST-{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}")
    

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

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    # class_distribution(training_data=training_data)

    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validate_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    train_dataset = train_dataloader.dataset

    labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    all_other_metric_reports = []

    ####### KFOLD STRATIFY
    skf = StratifiedKFold(
        n_splits=FOLDS, 
        shuffle=True, 
        random_state=RNG_SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, labels), start=1):
        print(f"Fold {fold}/{FOLDS}")


        # initiate all model related things
        model = NeuralNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
        
        # wandb initialize
        wandb.init(
            project="tutorial_cnn", 
            group=str(wandb_group_name),
            name=f"fold_{fold}", 
            config={
                "epoch": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": ALPHA,
                "num_folds": FOLDS,
                "model_structure": str(model)
            },
            reinit="finish_previous"
        )

        # split into train & test
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_kfold_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_kfold_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # epoch training runs
        epoch_train_losses = []
        epoch_train_accuracies = []
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_loop(
                dataloader=train_kfold_loader, 
                model=model, 
                loss_fn=loss_fn, 
                optimizer=optimizer
            )
            wandb.log({
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc
            })

            epoch_train_losses.append(train_loss)
            epoch_train_accuracies.append(train_acc)

        # test on kfold test set (not validation set)
        test_loss, test_acc, _, _, metrics_report = test_loop(
            dataloader=test_kfold_loader, 
            model=model, 
            loss_fn=loss_fn, 
            labels_map=labels_map
        )

        wandb.log({
            "fold": fold,
            "kfold/test_loss": test_loss,
            "kfold/test_accuracy": test_acc,
        })

        for class_name, class_metrics in metrics_report.items():
            # if class_name not in ["macro avg", "weighted avg", "accuracy"]:
            if isinstance(class_metrics, dict) and "precision" in class_metrics:
                wandb.log({
                    f"fold_{fold}_{class_name}_precision": class_metrics["precision"],
                    f"fold_{fold}_{class_name}_recall": class_metrics["recall"],
                    f"fold_{fold}_{class_name}_f1": class_metrics["f1-score"]
                })


        # impotent noombas
        all_train_losses.extend(epoch_train_losses)
        all_train_accuracies.extend(epoch_train_accuracies)
        all_test_losses.append(test_loss)
        all_test_accuracies.append(test_acc)
        all_other_metric_reports.append(metrics_report)

        wandb.finish()
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()



    ##### VALIDATATION
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)

    # epoch training runs
    epoch_train_losses = []
    epoch_train_accuracies = []
    for _ in range(EPOCHS):
        train_loss, train_acc = train_loop(
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn, 
            optimizer=optimizer
        )
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)

    # test on kfold test set (not validation set)
    val_loss, val_acc, _, _, metrics_report = test_loop(
        dataloader=validate_dataloader, 
        model=model, 
        loss_fn=loss_fn, 
        labels_map=labels_map
    )

    wandb.init(
        project="tutorial_cnn", 
        group=str(wandb_group_name),
        name=f"validation_set", 
        config={
            "epoch": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": ALPHA,
            "model_structure": str(model)
            # "num_folds": FOLDS,
        },
        reinit="finish_previous"
    )

    wandb.log({
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

    for class_name, class_metrics in metrics_report.items():
        if isinstance(class_metrics, dict) and "precision" in class_metrics:
            wandb.log({
                f"val_{class_name}_precision": class_metrics["precision"],
                f"val_{class_name}_recall": class_metrics["recall"],
                f"val_{class_name}_f1": class_metrics["f1-score"]
            })

    wandb.finish()

    # # Analysis
    # sample_output(
    #     model=model, 
    #     test_dataloader=test_dataloader, 
    #     labels_map=labels_map
    # )

    # loss_over_epoch(
    #     train_losses=train_losses,
    #     test_losses=test_losses,
    #     train_accuracies=train_accuracies,
    #     test_accuracies=test_accuracies
    # )

    # plot_confusion_matrix(
    #     test_labels=test_labels, 
    #     test_preds=test_preds, 
    #     labels_map=labels_map
    # )    

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time to run whole program: {time.time() - start_time}')