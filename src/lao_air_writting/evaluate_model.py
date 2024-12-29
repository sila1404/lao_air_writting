import time
from utils import CharacterRecognitionModel, VisualizationManager
import numpy as np
from sklearn.metrics import classification_report


def evaluate_model():
    # Use visualization methods
    viz = VisualizationManager()

    # Initialize model and load weights
    model = CharacterRecognitionModel()
    if not model.load_model():
        print("Failed to load model")
        return

    # Load test data
    print("Loading test data...")
    X_test, y_test, true_labels, label_map = model.load_test_data("augmented_images")

    # Make predictions
    print("\nMaking predictions...")
    start_time = time.time()
    predictions = model.model.predict(X_test)
    prediction_time = time.time() - start_time

    # Convert predictions to class labels
    y_pred = np.argmax(predictions, axis=1)

    # Calculate metrics
    print("\n=== Model Evaluation Results ===")
    print(f"\nPrediction time for {len(X_test)} samples: {prediction_time:.2f} seconds")
    print(
        f"Average prediction time per sample: {(prediction_time/len(X_test))*1000:.2f} ms"
    )

    # Print classification report
    label_names = list(label_map.keys())
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    viz.plot_confusion_matrix(y_test, y_pred, label_names)
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Load and plot training history
    try:
        # Load the history directly as a dictionary
        history_dict = np.load("training_history.npy", allow_pickle=True).item()

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

        print("\n=== Overfitting Analysis ===")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Accuracy Difference: {acc_diff:.4f}")

        if acc_diff > 0.1:
            print(
                "WARNING: Model shows signs of overfitting (accuracy difference > 0.1)"
            )
        else:
            print("Model shows no significant signs of overfitting")

    except Exception as e:
        print(f"\nCouldn't load training history: {e}")


if __name__ == "__main__":
    evaluate_model()
