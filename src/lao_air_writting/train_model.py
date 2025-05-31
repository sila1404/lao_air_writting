from utils import CharacterRecognitionModel
import os


def train_new_model():
    # Initialize model
    model = CharacterRecognitionModel()

    # Set the correct path to your dataset
    dataset_path = "train_datasets"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")

    try:
        # Train model
        print("Starting model training...")
        print(f"Using dataset from: {dataset_path}")

        history = model.train(data_dir=dataset_path, val_dir="val_datasets", epochs=20, batch_size=48)

        print("\nModel training completed!")

        return model, history

    except Exception as e:
        print(f"\nError during training: {str(e)}")


if __name__ == "__main__":
    train_new_model()
