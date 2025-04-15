import os
import random
import shutil

# Define paths
dataset_folder = "augmented_images"
output_train_folder = "train_datasets"
output_val_folder = "val_datasets"
output_test_folder = "test_datasets"

# Create output directories
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_val_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# Set the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Get all class subfolders
subfolders = [
    f
    for f in os.listdir(dataset_folder)
    if os.path.isdir(os.path.join(dataset_folder, f))
]
total_subfolders = len(subfolders)

# Track progress
completed_subfolders = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_folder, subfolder)

    # List all image files
    images = [
        f
        for f in os.listdir(subfolder_path)
        if os.path.isfile(os.path.join(subfolder_path, f))
    ]
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class subfolders in each dataset split
    os.makedirs(os.path.join(output_train_folder, subfolder), exist_ok=True)
    os.makedirs(os.path.join(output_val_folder, subfolder), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, subfolder), exist_ok=True)

    # Copy files
    for image in train_images:
        shutil.copy(
            os.path.join(subfolder_path, image),
            os.path.join(output_train_folder, subfolder, image),
        )

    for image in val_images:
        shutil.copy(
            os.path.join(subfolder_path, image),
            os.path.join(output_val_folder, subfolder, image),
        )

    for image in test_images:
        shutil.copy(
            os.path.join(subfolder_path, image),
            os.path.join(output_test_folder, subfolder, image),
        )

    completed_subfolders += 1
    print(f"Processed {completed_subfolders}/{total_subfolders} subfolders.")

print("Dataset split completed successfully.")
