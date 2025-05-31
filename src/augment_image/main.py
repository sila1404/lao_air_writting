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
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
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
        print(f"\nError during Keras {aug_type} for {base_name}: {str(e)}")


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
            augmented = transform(image=img)
            augmented_img = augmented["image"]
            output_filename = os.path.join(
                output_path, f"{base_name}_{aug_type}_{i + 1}.jpg"
            )
            Image.fromarray(augmented_img).save(output_filename)
    except Exception as e:
        print(f"\nError during Albumentations {aug_type} for {base_name}: {str(e)}")


def get_all_image_files(input_folder: str) -> List[str]:
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                # rel_path will be like "character_A/image1.jpg"
                rel_path = os.path.relpath(os.path.join(root, file), input_folder)
                image_files.append(rel_path)
    return image_files


def augment_images(input_folder: str, output_folder: str) -> None:
    image_files = get_all_image_files(input_folder)
    total_images = len(image_files)

    if not image_files:
        print(f"No JPG images found in '{input_folder}'")
        return

    print(f"Found {total_images} images to augment.")

    aug_iterations = {
        "rotated": 10,
        "width_shifted": 10,
        "height_shifted": 10,
        "zoomed": 10,
        "blur": 10,
        "motion_blur": 10,
        "perspective": 10,
    }

    keras_augmentations = create_keras_augmentations()
    albumentation_transforms = create_albumentation_transforms()

    os.makedirs(output_folder, exist_ok=True)

    processed_successfully = 0
    processed_with_errors = 0

    for idx, rel_path in enumerate(
        image_files
    ):  # rel_path is e.g., "FolderName/image_name.jpg"
        img_path = os.path.join(input_folder, rel_path)

        # Use rel_path directly in the progress message
        progress_message_base = f"Processing image {idx + 1}/{total_images}: {rel_path}"
        # Pad with spaces to clear previous longer messages. Adjust padding as needed.
        print(f"\r{progress_message_base:<90}", end="")

        output_subdir = os.path.join(output_folder, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        try:
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            original_output_path = os.path.join(
                output_subdir, f"{base_name}_original.jpg"
            )
            pil_img.save(original_output_path)

            img_array = img.reshape((1,) + img.shape)

            # Apply Keras augmentations
            for aug_type, datagen in keras_augmentations.items():
                print(f"\r{progress_message_base} - Keras: {aug_type:<15}", end="")
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
                print(
                    f"\r{progress_message_base} - Albumentations: {aug_type:<15}",
                    end="",
                )
                process_albumentation_image(
                    img,
                    transform,
                    output_subdir,
                    base_name,
                    aug_type,
                    aug_iterations[aug_type],
                )

            print(
                f"\r{progress_message_base} - Done. {'':<30}", end=""
            )  # Clear aug type part
            processed_successfully += 1

        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")
            processed_with_errors += 1
            # Reprint the base progress so the next image starts clean if an error occurred
            if idx + 1 < total_images:
                print(f"\r{progress_message_base:<90}", end="")

    print("\n\n🎉 Augmentation process completed!")
    print(f"Successfully augmented images: {processed_successfully}")
    if processed_with_errors > 0:
        print(f"Images with errors: {processed_with_errors}")
    print(f"Output saved to: {output_folder}")


if __name__ == "__main__":
    input_folder = "datasets"
    output_folder = "augmented_images"
    augment_images(input_folder, output_folder)
