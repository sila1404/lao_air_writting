import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras import regularizers # type: ignore
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
        l2_reg = regularizers.l2(0.001)
        
        model = models.Sequential(
            [
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2_reg, input_shape=(128, 128, 1)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2_reg),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2_reg),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                # Flatten and Dense Layers
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu", kernel_regularizer=l2_reg),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
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

        # Load data from train
        X_train, y_train, _, self.label_map = self.load_data_from_folder(
            data_dir, load_label_map=False
        )
        X_val, y_val, _, _ = self.load_data_from_folder("val_datasets")

        # Create and compile model
        num_classes = len(self.label_map)
        self.model = self.create_cnn_model(num_classes)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Prepare datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
            batch_size
        )

        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
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

    def load_data_from_folder(self, data_dir, img_size=(128, 128), load_label_map=True):
        """Load image data and optionally build or load label mapping."""
        images = []
        labels = []
        true_labels = []
        label_map = {}

        if load_label_map:
            with open("model/label_map.json", "r", encoding="utf-8") as f:
                label_map = json.load(f)
        else:
            # Dynamically create label map from folder names
            classes = sorted(
                [
                    d
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            )
            label_map = {class_name: idx for idx, class_name in enumerate(classes)}

        for class_name, label_idx in label_map.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, img_file)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, color_mode="grayscale", target_size=img_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0

                    images.append(img_array)
                    labels.append(label_idx)
                    true_labels.append(class_name)

        return np.array(images), np.array(labels), true_labels, label_map
