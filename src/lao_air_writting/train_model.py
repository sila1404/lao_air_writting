from utils import CharacterRecognitionModel
import os


def train_new_model():
    # Initialize model
    model = CharacterRecognitionModel()

    # Set the correct path to your dataset
    dataset_path = "augmented_images"  # Update this path if needed

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")

    try:
        # Train model
        print("Starting model training...")
        print(f"Using dataset from: {dataset_path}")

        history = model.train(data_dir=dataset_path, epochs=10, batch_size=32)

        print("\nModel training completed!")
        print("Files saved:")
        print("- hand_drawn_character_model.h5")
        print("- label_map.json")
        print("- training_history.npy")

        return model, history

    except Exception as e:
        print(f"\nError during training: {str(e)}")


if __name__ == "__main__":
    train_new_model()