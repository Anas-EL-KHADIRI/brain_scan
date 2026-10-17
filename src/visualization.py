from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import CLASS_NAMES, REPORTS_DIR


def plot_class_distribution(y, save_path=None):
    # Bar plot of dataset class distribution (Notebook Cell 8)
    unique_classes, class_counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique_classes, class_counts)
    plt.title("BrainScan AI - Class Distribution")
    plt.xlabel("Brain MRI Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def show_sample_images(X, y, class_names=CLASS_NAMES, save_path=None):
    # Grid of sample images per class (Notebook Cell 11)
    plt.figure(figsize=(12, 6))
    for idx, class_name in enumerate(class_names):
        class_idx = np.where(y == class_name)[0][0]

        plt.subplot(1, 4, idx + 1)
        plt.imshow(X[class_idx])
        plt.title(f"{class_name.capitalize()}")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path=None):
    # Confusion matrix display (Notebook Cell 21)
    cm = confusion_matrix(
        y_true, y_pred,
        labels=np.arange(len(class_names))
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Brain Tumor Classification - Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    return cm


def plot_predictions(X_test, y_true, y_pred, class_names=CLASS_NAMES, indices=None, title="Predictions", save_path=None):
    # Plot prediction samples with True vs Predicted labels (Notebook Cell 22)
    if indices is None:
        indices = np.arange(min(8, len(X_test)))

    plt.figure(figsize=(14, 7))
    for i, idx in enumerate(indices[:8]):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X_test[idx])

        actual = class_names[y_true[idx]]
        predicted = class_names[y_pred[idx]]

        plt.title(f"True: {actual}\nPred: {predicted}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_learning_curves(history, save_path=None):
    # Plot training and validation loss and accuracy curves (Notebook Cell 18)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history.get("loss", []), label="Train Loss", color="royalblue")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="Val Loss", color="darkorange")
    ax1.set_title("BrainScan AI - Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    ax2.plot(history.history.get("accuracy", []), label="Train Accuracy", color="royalblue")
    if "val_accuracy" in history.history:
        ax2.plot(history.history["val_accuracy"], label="Val Accuracy", color="darkorange")
    ax2.set_title("BrainScan AI - Accuracy Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
