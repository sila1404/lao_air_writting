from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
import albumentations as A
from typing import Dict, List


def create_keras_augmentations() -> Dict[str, ImageDataGenerator]:
    return {
        "rotated": ImageDataGenerator(rotation_range=30),
        "width_shifted": ImageDataGenerator(width_shift_range=0.2),
        "height_shifted": ImageDataGenerator(height_shift_range=0.2),
        "zoomed": ImageDataGenerator(zoom_range=0.2),
    }


def create_albumentation_transforms() -> Dict[str, A.Compose]:
    return {
        "blur": A.Compose(
            [
                A.Blur(blur_limit=(3, 7), p=1.0),
            ]
        ),
        "motion_blur": A.Compose(
            [
                A.MotionBlur(blur_limit=(3, 8), p=1.0),
            ]
        ),
        "perspective": A.Compose(
            [
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ]
        ),
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


def process_albumentation_image(
    img: np.ndarray,
    transform: A.Compose,
    output_path: str,
    base_name: str,
    aug_type: str,
    iterations: int = 1,
) -> None:
    try:
        for i in range(iterations):
            # Apply the transformation
            augmented = transform(image=img)
            augmented_img = augmented["image"]

            # Convert to PIL Image and save
            output_filename = os.path.join(
                output_path, f"{base_name}_{aug_type}_{i + 1}.jpg"
            )
            Image.fromarray(augmented_img).save(output_filename)
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

    # Define augmentation iterations
    aug_iterations = {
        # Keras augmentations
        "rotated": 10,
        "width_shifted": 10,
        "height_shifted": 10,
        "zoomed": 10,
        # Albumentations augmentations
        "blur": 10,
        "motion_blur": 10,
        "perspective": 10,
    }

    # Create augmentation generators/transforms
    keras_augmentations = create_keras_augmentations()
    albumentation_transforms = create_albumentation_transforms()

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

            # Prepare for Keras augmentation
            img_array = img.reshape((1,) + img.shape)

            # Apply Keras augmentations
            for aug_type, datagen in keras_augmentations.items():
                print(f"  Applying {aug_type}...")
                process_single_image(
                    img_array,
                    datagen,
                    output_subdir,
                    base_name,
                    aug_type,
                    aug_iterations[aug_type],
                )

            # Apply Albumentations transforms
            for aug_type, transform in albumentation_transforms.items():
                print(f"  Applying {aug_type}...")
                process_albumentation_image(
                    img,
                    transform,
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
