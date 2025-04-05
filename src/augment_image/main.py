from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2
import os
from typing import Dict, List
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
    try:
        for i in range(iterations):
            next(
                datagen.flow(
                    img_array,
                    batch_size=1,
                    save_to_dir=output_path,
                    save_prefix=f"{base_name}_{aug_type}_{i + 1}",
                    save_format="jpg",
                )
            )
    except Exception as e:
        print(f"Error processing augmentation {aug_type} for {base_name}: {str(e)}")


def get_all_image_files(input_folder: str) -> List[str]:
    """Recursively get all jpg images from input folder and its subdirectories"""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist")

    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):  # Added .jpeg extension
                # Store full path relative to input folder
                rel_path = os.path.relpath(root, input_folder)
                if rel_path == ".":
                    image_files.append(file)
                else:
                    image_files.append(os.path.join(rel_path, file))

    return image_files


def augment_images(input_folder: str, output_folder: str) -> None:
    """Main function to handle image augmentation"""
    try:
        # Get all jpg images including those in subdirectories
        image_files = get_all_image_files(input_folder)

        if not image_files:
            print(f"No JPG images found in '{input_folder}' or its subdirectories")
            return

        print(f"Found {len(image_files)} images")
        print("Images found in following locations:")
        for img in image_files:
            print(f"  - {os.path.join(input_folder, img)}")

        # Define augmentation iterations
        aug_iterations = {
            "rotated": 40,
            "width_shifted": 20,
            "height_shifted": 20,
            "zoomed": 20,
        }

        # Get augmentation configurations
        augmentations = create_augmentations()

        # Create base output directory
        os.makedirs(output_folder, exist_ok=True)

        for image_file in image_files:
            print(f"\nProcessing {image_file}...")

            # Create output subdirectory structure if needed
            output_subdir = os.path.join(output_folder, os.path.dirname(image_file))
            os.makedirs(output_subdir, exist_ok=True)

            # Load and preprocess image
            img_path = os.path.join(input_folder, image_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                base_name = os.path.splitext(os.path.basename(image_file))[0]

                # Save original image
                original_output_path = os.path.join(
                    output_subdir, f"{base_name}_original.jpg"
                )
                cv2.imwrite(
                    original_output_path,
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )

                # Reshape for augmentation
                img_array = img.reshape((1,) + img.shape)

                # Apply all augmentations
                for aug_type, datagen in augmentations.items():
                    print(f"  Applying {aug_type}...")
                    process_single_image(
                        img_array,
                        datagen,
                        output_subdir,
                        base_name,
                        aug_type,
                        aug_iterations[aug_type],
                    )

                print(f"Completed augmentations for {image_file}")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        print("\nAugmentation process completed!")
        print(f"Output files can be found in: {output_folder}")

    except Exception as e:
        print(f"An error occurred during the augmentation process: {str(e)}")


if __name__ == "__main__":
    input_folder = "datasets"
    output_folder = "augmented_images"
    augment_images(input_folder, output_folder)
