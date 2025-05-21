import time
from utils import CharacterRecognitionModel, VisualizationManager
import numpy as np
from sklearn.metrics import classification_report
import json
import os
import math


def evaluate_model():
    # Use visualization methods
    viz = VisualizationManager()

    # Initialize model and load weights
    model = CharacterRecognitionModel()
    if not model.load_model(use_quantize_model=True):
        print("Failed to load model")
    else:
        print("Model loaded")

    # Load test data
    print("\nLoading test data... This might take a moment.")
    test_dataset, true_labels, label_map = model.load_data_from_folder(
        "test_datasets", load_label_map=True, in_memory=True
    )
    num_test_samples = len(true_labels)
    if num_test_samples == 0:
        print("No test samples found in 'test_datasets'. Evaluation cannot proceed.")
    print(f"Test data loaded. Found {num_test_samples} samples.")

    # Batch the dataset
    batch_size = 32  # Define batch_size here to use for progress calculation
    test_dataset = test_dataset.batch(batch_size)
    num_batches = math.ceil(num_test_samples / batch_size)

    y_test = []
    y_pred = []

    print("\nMaking predictions...")
    start_time = time.time()
    processed_samples = 0

    for i, (images, labels) in enumerate(test_dataset):
        predictions = model.model.predict(images, verbose=0)
        y_pred_batch = np.argmax(predictions, axis=1)
        y_pred.extend(y_pred_batch)
        y_test.extend(labels.numpy())

        processed_samples += len(labels.numpy())
        # Update progress on the same line
        print(
            f"Processed batch {i + 1}/{num_batches} ({processed_samples}/{num_test_samples} samples)",
            end="\r",
        )

    # Print a newline after the loop to move to the next line
    print(
        "\nPredictions completed.                                       "
    )  # Extra spaces to clear the line

    prediction_time = time.time() - start_time

    # Calculate metrics
    print("\n=== Model Evaluation Results ===")
    if (
        not y_test
    ):  # Should have been caught by num_test_samples check, but good for safety
        print("No test samples were processed to evaluate.")

    print(f"\nPrediction time for {len(y_test)} samples: {prediction_time:.2f} seconds")
    print(
        f"Average prediction time per sample: {(prediction_time / len(y_test)) * 1000:.2f} ms"
    )

    # Print classification report
    label_names = list(label_map.keys())
    report_path = "eval/evaluation_report.txt"
    print(f"\nClassification Report (saved to {report_path}):")

    os.makedirs("eval", exist_ok=True)

    classification_output = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_output)
    print(f"\nClassification Report has been saved to: {report_path}")

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    viz.plot_confusion_matrix(y_test, y_pred, label_names)
    print("Confusion matrix saved as 'eval/confusion_matrix.png'")

    # Load and plot training history
    history_path = "model/training_history.json"
    print(f"\nLoading training history from '{history_path}'...")
    try:
        with open(history_path, "r") as f:
            history_dict = json.load(f)

        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict

        history_for_plot = HistoryWrapper(history_dict)

        viz.plot_training_history(history_for_plot)
        print("Training history plot saved as 'eval/training_history.png'")

        # Analyze overfitting
        if (
            "accuracy" in history_dict
            and "val_accuracy" in history_dict
            and history_dict["accuracy"]
            and history_dict["val_accuracy"]
        ):
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
                print(
                    "Model shows no significant signs of overfitting or underfitting."
                )
        else:
            print(
                "\nCould not perform generalization analysis: 'accuracy' or 'val_accuracy' missing or empty in training history."
            )

    except FileNotFoundError:
        print(f"\nCouldn't load training history: File not found at '{history_path}'.")
    except Exception as e:
        print(f"\nError processing training history: {e}")


if __name__ == "__main__":
    evaluate_model()
