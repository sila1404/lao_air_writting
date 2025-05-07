import tensorflow as tf
import keras
from keras import layers, models, Input
import cv2
import numpy as np
import os
import json
from typing import Optional, Dict, Tuple, List


class CharacterRecognitionModel:
    def __init__(self):
        """
        Initializes an empty CharacterRecognitionModel instance.
        """
        self.model: Optional[keras.Model] = None
        self.label_map: Optional[Dict[str, int]] = None
        self.history: Optional[keras.callbacks.History] = None

    def create_cnn_model(self, num_classes: int) -> keras.Model:
        """
        Creates a CNN model for character recognition.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            keras.Model: Compiled CNN model.
        """

        model = models.Sequential(
            [
                Input(shape=(128, 128, 1)),
                # First Convolutional Block
                layers.Conv2D(64, (5, 5), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Second Convolutional Block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Third Convolutional Block
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Forth Convolutional Block
                layers.Conv2D(512, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Flatten and Dense Layers
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model

    def train(
        self,
        data_dir: str,
        val_dir: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Trains the CNN model with TensorBoard logging.

        Args:
            data_dir (str): Path to the training dataset directory.
            val_dir (Optional[str], optional): Path to validation dataset. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 50.
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            keras.callbacks.History: Training history object.
        """

        # Ensure TensorFlow uses GPU
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            print(f"GPU devices available: {len(physical_devices)}")
            # Set memory growth to avoid consuming all GPU memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Using GPU for computations")
        else:
            print("No GPU found. Using CPU instead.")

        # Load training datasets
        train_dataset, _, self.label_map = self.load_data_from_folder(
            data_dir, load_label_map=False, shuffle_data=True
        )
        train_dataset = train_dataset.batch(batch_size)

        # Save label map early so validation can use it
        os.makedirs("model", exist_ok=True)
        with open("model/label_map.json", "w", encoding="utf-8") as f:
            json.dump(self.label_map, f, ensure_ascii=False, indent=2)

        # Load validation datasets only if provided
        val_dataset = None
        callbacks = []
        if val_dir:
            val_dataset, _, _ = self.load_data_from_folder(val_dir, load_label_map=True)
            val_dataset = val_dataset.batch(batch_size)

            callbacks.extend(
                [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=28, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.2,  # Reduce learning rate by a factor of 20
                        patience=9,  # Reduce if val_loss doesn't improve for 9 epochs
                        min_lr=3e-7,  # Minimum learning rate
                        verbose=1,
                    ),
                ]
            )

        # Create and compile model
        num_classes = len(self.label_map)
        self.model = self.create_cnn_model(num_classes)
        optimizer = keras.optimizers.AdamW(learning_rate=5e-3, weight_decay=1e-3)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset if val_dataset else None,
            callbacks=callbacks,
        )

        # Save model and label mapping
        self.save_model()

        return self.history

    def save_model(
        self,
        model_path: str = "model/hand_drawn_character_model.keras",
        history_path: str = "model/training_history.json",
    ):
        """
        Saves the trained model and training history to disk.

        Args:
            model_path (str, optional): Path to save the model file.
            history_path (str, optional): Path to save training history as JSON.
        """

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path, save_format="h5")

        # Save training history as JSON
        if self.history is not None:
            # Convert any NumPy types to native Python types
            history_data = {}
            for key, values in self.history.history.items():
                history_data[key] = [
                    float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for v in values
                ]

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2)

    def load_model(
        self,
        model_path: str = "model/hand_drawn_character_model.keras",
        label_map_path: str = "model/label_map.json",
    ) -> bool:
        """
        Loads a trained model and corresponding label map.

        Args:
            model_path (str, optional): Path to model file.
            label_map_path (str, optional): Path to label map JSON.

        Returns:
            bool: True if loading is successful, False otherwise.
        """

        try:
            self.model = keras.models.load_model(model_path)
            # Load label map with proper Unicode encoding
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, canvas: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predicts the character from a given image.

        Args:
            canvas (np.ndarray): Input image array (grayscale or BGR).

        Returns:
            Tuple[Optional[str], float]: Predicted label and confidence score.
        """

        if self.model is None or self.label_map is None:
            return None, 0.0

        # Ensure the canvas is in grayscale (single channel)
        if len(canvas.shape) == 3:  # If it's a 3-channel image (BGR)
            img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:  # If it's already grayscale
            img = canvas

        # Resize the image to match the model's input size
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # Normalize and reshape the image for the model
        img = img / 255.0
        img = np.expand_dims(img, axis=[0, -1])

        # Make prediction
        prediction = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction[0])

        # Map the predicted class to the corresponding label
        reverse_label_map = {str(v): k for k, v in self.label_map.items()}
        predicted_label = reverse_label_map[str(predicted_class)]
        confidence = prediction[0][predicted_class]

        return predicted_label, confidence

    def load_data_from_folder(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (128, 128),
        load_label_map: bool = True,
        shuffle_data: bool = False,
        in_memory: bool = False,
    ) -> Tuple[tf.data.Dataset, List[str], Dict[str, int]]:
        """
        Loads image dataset from a folder structure.

        Args:
            data_dir (str): Path to the dataset directory.
            img_size (Tuple[int, int], optional): Size to resize images. Defaults to (128, 128).
            load_label_map (bool, optional): Whether to load existing label map. Defaults to True.
            shuffle_data (bool, optional): Whether to shuffle dataset. Defaults to False.
            in_memory (bool, optional): Whether to fully cache dataset in memory. Defaults to False.

        Returns:
            Tuple[tf.data.Dataset, List[str], Dict[str, int]]:
                Dataset object, true labels list, and label mapping.
        """

        def process_path(file_path, label):
            # Load and preprocess image
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(
                img, channels=1, expand_animations=False
            )  # Grayscale
            img = tf.image.resize(img, img_size, method="area")
            img = img / 255.0  # Normalize
            return img, label

        # Load or create label map
        if load_label_map:
            with open("model/label_map.json", "r", encoding="utf-8") as f:
                label_map = json.load(f)
        else:
            classes = sorted(
                [
                    d
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            )
            label_map = {class_name: idx for idx, class_name in enumerate(classes)}

        # Gather file paths and labels
        file_paths = []
        labels = []
        true_labels = []
        for class_name, label_idx in label_map.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_paths.append(os.path.join(class_path, img_file))
                    labels.append(label_idx)
                    true_labels.append(class_name)

        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        if in_memory:
            dataset = dataset.cache()
        else:
            os.makedirs("cache", exist_ok=True)
            cache_path = (
                "cache/train_cache.tf-data"
                if shuffle_data
                else "cache/val_cache.tf-data"
            )
            dataset = dataset.cache(cache_path)
            for _ in dataset:  # Forces full traversal so TF writes the entire cache
                pass

        if shuffle_data:
            dataset = dataset.shuffle(1000)  # Shuffle only if requested

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, true_labels, label_map
