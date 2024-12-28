from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2
import os
from typing import Dict
import numpy as np


def create_augmentations() -> Dict[str, ImageDataGenerator]:
    """Define all augmentation configurations"""
    return {
        "rotated": ImageDataGenerator(rotation_range=40),
        "width_shifted": ImageDataGenerator(width_shift_range=0.2),
        "height_shifted": ImageDataGenerator(height_shift_range=0.2),
        "zoomed": ImageDataGenerator(zoom_range=0.2),
    }


def process_single_image(
    img_array: np.ndarray,
    datagen: ImageDataGenerator,
    output_path: str,
    base_name: str,
    aug_type: str,
    iterations: int = 1,
) -> None:
    """Process a single image with given augmentation settings"""
    for i in range(iterations):
        next(
            datagen.flow(
                img_array,
                batch_size=1,
                save_to_dir=output_path,
                save_prefix=f"{base_name}_{aug_type}_{i+1}",
                save_format="jpg",
            )
        )


def augment_images(input_folder: str, output_folder: str) -> None:
    """Main function to handle image augmentation"""
    # Create output directory if needed
    os.makedirs(output_folder, exist_ok=True)

    # Get all jpg images
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
    print(f"Found {len(image_files)} images")

    # Define augmentation iterations
    aug_iterations = {
        "rotated": 40,
        "width_shifted": 20,
        "height_shifted": 20,
        "zoomed": 20,
    }

    # Get augmentation configurations
    augmentations = create_augmentations()

    for image_file in image_files:
        print(f"Processing {image_file}...")

        # Load and preprocess image
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base_name = os.path.splitext(image_file)[0]

        # Save original image
        cv2.imwrite(
            os.path.join(output_folder, f"{base_name}_original.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

        # Reshape for augmentation
        img_array = img.reshape((1,) + img.shape)

        # Apply all augmentations
        for aug_type, datagen in augmentations.items():
            print(f"Applying {aug_type}...")
            process_single_image(
                img_array,
                datagen,
                output_folder,
                base_name,
                aug_type,
                aug_iterations[aug_type],
            )

        print(f"Completed augmentations for {image_file}")


if __name__ == "__main__":
    input_folder = "datasets"
    output_folder = "augmented_images"
    augment_images(input_folder, output_folder)
