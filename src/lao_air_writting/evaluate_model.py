import time
from utils import CharacterRecognitionModel, VisualizationManager
import numpy as np
from sklearn.metrics import classification_report
import json
import os


def evaluate_model():
    # Use visualization methods
    viz = VisualizationManager()

    # Initialize model and load weights
    model = CharacterRecognitionModel()
    if not model.load_model():
        print("Failed to load model")
    else:
        print("Model loaded")

    # Load test data
    print("Loading test data...")
    test_dataset, true_labels, label_map = model.load_data_from_folder("test_datasets")

    # Instead of looping, let's directly use `test_dataset` and batch the data
    test_dataset = test_dataset.batch(
        32
    )  # Adjust batch size based on your available memory

    # Initialize empty lists to store predictions and true labels
    y_test = []
    y_pred = []

    # Make predictions in batches
    print("\nMaking predictions...")
    start_time = time.time()

    # Loop through the test dataset and accumulate predictions
    for images, labels in test_dataset:
        # Make predictions for the batch
        predictions = model.model.predict(images)

        # Convert predictions to class labels and store them
        y_pred_batch = np.argmax(predictions, axis=1)
        y_pred.extend(y_pred_batch)

        # Append true labels (convert TensorFlow tensor to NumPy)
        y_test.extend(labels.numpy())

    # Calculate prediction time
    prediction_time = time.time() - start_time

    # Calculate metrics
    print("\n=== Model Evaluation Results ===")
    print(f"\nPrediction time for {len(y_test)} samples: {prediction_time:.2f} seconds")
    print(
        f"Average prediction time per sample: {(prediction_time / len(y_test)) * 1000:.2f} ms"
    )

    # Print classification report
    label_names = list(label_map.keys())
    print("\nClassification Report:")
    os.makedirs("eval", exist_ok=True)
    report_path = "eval/evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                y_test, y_pred, target_names=label_names, zero_division=0
            )
        )
        print(f"\nClassification Report has been saved to: {report_path}")

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    viz.plot_confusion_matrix(y_test, y_pred, label_names)
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Load and plot training history
    try:
        # Load the history directly as a dictionary
        history_dict = json.load(open("model/training_history.json"))

        # Create a simple object to match the plotting function's expectations
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        history = HistoryWrapper(history_dict)

        viz.plot_training_history(history)
        print("Training history plot saved as 'training_history.png'")

        # Analyze overfitting
        final_train_acc = history_dict["accuracy"][-1]
        final_val_acc = history_dict["val_accuracy"][-1]
        acc_diff = final_train_acc - final_val_acc

        print("\n=== Model Generalization Analysis ===")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Accuracy Difference: {acc_diff:.4f}")

        if final_train_acc < 0.5 and final_val_acc < 0.5:
            print(
                "WARNING: Model shows signs of underfitting (low accuracy on both training and validation)"
            )
        elif acc_diff > 0.1:
            print(
                "WARNING: Model shows signs of overfitting (accuracy difference > 0.1)"
            )
        else:
            print("Model shows no significant signs of overfitting or underfitting")

    except Exception as e:
        print(f"\nCouldn't load training history: {e}")


if __name__ == "__main__":
    evaluate_model()
