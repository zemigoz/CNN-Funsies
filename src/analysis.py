from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

from sklearn.metrics import confusion_matrix



def loss_over_epoch(train_losses, test_losses, train_accuracies, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(test_losses, label='Test Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accuracies, label='Train Accuracy', marker='o')
    ax2.plot(test_accuracies, label='Test Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(test_labels, test_preds, labels_map):
    cm = confusion_matrix(test_labels, test_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels_map.values(),
        yticklabels=labels_map.values()
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    class_accuracy = class_correct / class_total

    print("\nPer-Class Accuracy:")
    for idx, (name, acc) in enumerate(zip(labels_map.values(), class_accuracy)):
        print(f"{name:12s}: {acc:.2%}")


def class_distribution(training_data):
    labels = [label for _, label in training_data]
    label_counts = Counter(labels)

    print("Class distribution:")
    for class_id, count in sorted(label_counts.items()):
        print(f"Class {class_id}: {count} samples")

    
    total = len(labels)
    print(f"\nTotal samples: {total}")
    for class_id, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
    print(f"Class {class_id}: {percentage:.1f}%")


def sample_output(model, test_dataloader, labels_map):
    test_images, test_labels = next(iter(test_dataloader))
    logits = model(test_images)

    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    
    print(f"Predicted class: {y_pred}")

    # plot 9 images and their predicted/actual labels
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(min(cols * rows, len(test_images))):
        figure.add_subplot(rows, cols, i + 1)
        
        pred_label = labels_map[y_pred[i].item()]
        true_label = labels_map[test_labels[i].item()]
        
        # green if correct, red wrong
        color = 'green' if y_pred[i] == test_labels[i] else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        
        plt.axis("off")
        plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.show()