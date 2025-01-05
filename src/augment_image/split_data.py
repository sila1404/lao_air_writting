import os
import random
import shutil

# Define paths
dataset_folder = "augment_image"
output_test_folder = "test_datasets"
output_train_folder = "train_datasets"

# Create output directories
os.makedirs(output_test_folder, exist_ok=True)
os.makedirs(output_train_folder, exist_ok=True)

# Set the split ratio
test_ratio = 0.2

# Get all subfolders
subfolders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
total_subfolders = len(subfolders)

# Initialize counter for tracking progress
completed_subfolders = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_folder, subfolder)

    # List all images in the subfolder
    images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

    # Shuffle and split images
    random.shuffle(images)
    split_index = int(len(images) * test_ratio)
    test_images = images[:split_index]
    train_images = images[split_index:]

    # Create corresponding subfolders in output directories
    os.makedirs(os.path.join(output_test_folder, subfolder), exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, subfolder), exist_ok=True)

    # Move test images
    for image in test_images:
        shutil.copy(os.path.join(subfolder_path, image), os.path.join(output_test_folder, subfolder, image))

    # Move train images
    for image in train_images:
        shutil.copy(os.path.join(subfolder_path, image), os.path.join(output_train_folder, subfolder, image))

    # Update progress
    completed_subfolders += 1
    print(f"Processed {completed_subfolders}/{total_subfolders} subfolders.")

print("Dataset split completed.")
