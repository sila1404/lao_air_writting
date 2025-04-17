from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
from typing import Dict, List


def create_augmentations() -> Dict[str, ImageDataGenerator]:
    return {
        "rotated": ImageDataGenerator(rotation_range=40),
        "width_shifted": ImageDataGenerator(width_shift_range=0.2),
        "height_shifted": ImageDataGenerator(height_shift_range=0.2),
        "zoomed": ImageDataGenerator(zoom_range=0.2),
        "sheared": ImageDataGenerator(shear_range=0.2),
    }


def process_single_image(
    img_array: np.ndarray,
    datagen: ImageDataGenerator,
    output_path: str,
    base_name: str,
    aug_type: str,
    iterations: int = 1,
) -> None:
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
        print(f"Error processing {aug_type} for {base_name}: {str(e)}")


def get_all_image_files(input_folder: str) -> List[str]:
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                rel_path = os.path.relpath(os.path.join(root, file), input_folder)
                image_files.append(rel_path)
    return image_files


def augment_images(input_folder: str, output_folder: str) -> None:
    image_files = get_all_image_files(input_folder)

    if not image_files:
        print(f"No JPG images found in '{input_folder}'")
        return

    print(f"Found {len(image_files)} images.")

    aug_iterations = {
        "rotated": 10,
        "width_shifted": 10,
        "height_shifted": 10,
        "zoomed": 10,
        "sheared": 10,
    }

    augmentations = create_augmentations()
    os.makedirs(output_folder, exist_ok=True)

    for rel_path in image_files:
        img_path = os.path.join(input_folder, rel_path)
        output_subdir = os.path.join(output_folder, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        try:
            print(f"\nProcessing {img_path}...")

            # Use Pillow to open image
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)

            # Save original image
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            original_output_path = os.path.join(
                output_subdir, f"{base_name}_original.jpg"
            )
            pil_img.save(original_output_path)

            # Prepare for augmentation
            img_array = img.reshape((1,) + img.shape)

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

            print(f"✅ Completed augmentations for: {img_path}")

        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")
            continue

    print("\n🎉 Augmentation process completed!")
    print(f"Output saved to: {output_folder}")


if __name__ == "__main__":
    input_folder = "datasets"
    output_folder = "augmented_images"
    augment_images(input_folder, output_folder)
