import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
import cv2
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from pathlib import Path


class CharacterRecognitionModel:
    def __init__(self):
        self.model = None
        self.label_map = None
        self.history = None

    def load_and_preprocess_data(self, data_dir, img_size=(128, 128)):
        images = []
        labels = []
        label_map = {}
        current_label = 0

        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' not found!")

        class_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(data_dir / d)]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")

        print(f"Found {len(class_dirs)} classes: {class_dirs}")

        for class_name in class_dirs:
            class_path = data_dir / class_name

            # Add class to label map
            label_map[class_name] = current_label
            print(f"Processing class '{class_name}' (label: {current_label})")

            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"Warning: No images found in class directory '{class_name}'")
                continue

            print(f"Found {len(image_files)} images in class '{class_name}'")

            for image_file in image_files:
                img_path = str(class_path / image_file)
                try:
                    # Read image using numpy and cv2.imdecode
                    img_array = np.fromfile(img_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f"Warning: Could not read image {image_file}")
                        continue

                    # Resize image if needed
                    if img.shape != img_size:
                        img = cv2.resize(img, img_size)

                    # Normalize and add channel dimension
                    img = img / 255.0
                    img = np.expand_dims(img, axis=-1)

                    images.append(img)
                    labels.append(current_label)
                except Exception as e:
                    print(f"Error processing image {image_file}: {str(e)}")
                    continue

            current_label += 1

        if not images:
            raise ValueError("No valid images were loaded!")

        print("\nDataset summary:")
        print(f"Total images loaded: {len(images)}")
        print(f"Number of classes: {len(label_map)}")
        print(f"Label mapping: {label_map}")

        return np.array(images), np.array(labels), label_map

    def create_cnn_model(self, num_classes):
        model = models.Sequential(
            [
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Flatten and Dense Layers
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model

    def train(self, data_dir, epochs=50, batch_size=32):
        # Load and preprocess data
        images, labels, self.label_map = self.load_and_preprocess_data(data_dir)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        # Create and compile model
        num_classes = len(self.label_map)
        self.model = self.create_cnn_model(num_classes)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Data augmentation
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.1, 0.1),
            ]
        )

        # Prepare datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
            batch_size
        )

        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=test_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
        )

        # Save model and label mapping
        self.save_model()

        return self.history

    def save_model(
        self,
        model_path="hand_drawn_character_model.h5",
        label_map_path="label_map.json",
    ):
        self.model.save(model_path)
        # Save label map with proper Unicode encoding
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(self.label_map, f, ensure_ascii=False, indent=2)

    def load_model(
        self,
        model_path="hand_drawn_character_model.h5",
        label_map_path="label_map.json",
    ):
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Load label map with proper Unicode encoding
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, canvas):
        if self.model is None or self.label_map is None:
            return None, 0.0

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = np.expand_dims(img, axis=[0, -1])

        prediction = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction[0])

        reverse_label_map = {str(v): k for k, v in self.label_map.items()}
        predicted_label = reverse_label_map[str(predicted_class)]
        confidence = prediction[0][predicted_class]

        return predicted_label, confidence

    def evaluate(self, test_dataset):
        if self.model is None:
            print("Model not loaded")
            return

        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")