import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class VisualizationManager:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        # Ensure the eval directory exists
        os.makedirs("eval", exist_ok=True)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig("eval/confusion_matrix.png")
        plt.close()

    @staticmethod
    def plot_training_history(history):
        """Plot training history"""

        # Ensure the eval directory exists
        os.makedirs("eval", exist_ok=True)

        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig("eval/training_history.png")
        plt.close()
