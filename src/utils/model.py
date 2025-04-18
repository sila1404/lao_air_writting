import tensorflow as tf
from keras import layers, models, Input
import cv2
import numpy as np
import os
import json


class CharacterRecognitionModel:
    def __init__(self):
        self.model = None
        self.label_map = None
        self.history = None

    def create_cnn_model(self, num_classes):
        model = models.Sequential(
            [
                Input(shape=(128, 128, 1)),
                # First Convolutional Block
                layers.Conv2D(64, (5, 5), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # First Convolutional Block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Second Convolutional Block
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Flatten and Dense Layers
                layers.Flatten(),
                layers.Dropout(0.4),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        return model

    def train(self, data_dir, epochs=50, batch_size=32):
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

        # Load datasets
        train_dataset, _, self.label_map = self.load_data_from_folder(
            data_dir, load_label_map=False, shuffle_data=True
        )
        val_dataset, _, _ = self.load_data_from_folder(
            "val_datasets", load_label_map=True
        )

        # Batch datasets
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        # Create and compile model
        num_classes = len(self.label_map)
        self.model = self.create_cnn_model(num_classes)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-3, weight_decay=4e-3
        )  # 0.001
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,  # Reduce learning rate by a factor of 5
                    patience=4,  # Reduce if val_loss doesn't improve for 4 epochs
                    min_lr=1e-6,  # Minimum learning rate
                    verbose=1,
                ),
            ],
        )

        # Save model and label mapping
        self.save_model()

        return self.history

    def save_model(
        self,
        model_path="model/hand_drawn_character_model.keras",
        label_map_path="model/label_map.json",
        history_path="model/training_history.json",
    ):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            self.model.save(model_path)

        # Save label map
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(self.label_map, f, ensure_ascii=False, indent=2)

        # Save training history as JSON
        if self.history is not None:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history.history, f, indent=2)

    def load_model(
        self,
        model_path="model/hand_drawn_character_model.keras",
        label_map_path="model/label_map.json",
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
        self, data_dir, img_size=(128, 128), load_label_map=True, shuffle_data=False
    ):
        """Load image data using tf.data for efficient loading and preprocessing."""

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
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).cache()

        if shuffle_data:
            dataset = dataset.shuffle(1000)  # Shuffle only if requested

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, true_labels, label_map
