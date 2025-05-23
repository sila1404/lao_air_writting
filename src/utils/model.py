import tensorflow as tf
import keras
from keras import layers, models, Input
import cv2
import numpy as np
import os
import json
from typing import Optional, Dict, Tuple, List, Any


class CharacterRecognitionModel:
    def __init__(self):
        """
        Initializes an empty CharacterRecognitionModel instance.
        """
        self.model: Optional[keras.Model] = None
        self.label_map: Optional[Dict[str, int]] = None
        self.history: Optional[keras.callbacks.History] = None
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.input_details: Optional[List[Dict[str, Any]]] = None
        self.output_details: Optional[List[Dict[str, Any]]] = None

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
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.6),
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
                        monitor="val_loss", patience=16, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.2,  # Reduce learning rate by a factor of 20
                        patience=7,  # Reduce if val_loss doesn't improve for 7 epochs
                        min_lr=1e-6,  # Minimum learning rate
                        verbose=1,
                    ),
                ]
            )

        # Create and compile model
        num_classes = len(self.label_map)
        self.model = self.create_cnn_model(num_classes)
        optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2)
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

        self.convert_to_tflite()

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

    def convert_to_tflite(
        self,
        model_path: str = "model/hand_drawn_character_model.keras",
        quantized_path: str = "model/hand_drawn_character_model_quant.tflite",
    ):
        """
        Converts the Keras model to a quantized TFLite model.

        Args:
            model_path (str, optional): Path to the Keras model file.
            quantized_path (str, optional): Path to save the quantized TFLite model.
        """
        if not os.path.exists(model_path):
            print(f" Error: Model file not found at {model_path}")
            return

        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded Keras model from: {model_path}")

            original_size = os.path.getsize(model_path) / (1024 * 1024)

            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Key changes for float16 quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            print("Starting TFLite model conversion with float16 quantization...")
            tflite_model_quantized = converter.convert()
            print("Model conversion successful.")

            os.makedirs(os.path.dirname(quantized_path), exist_ok=True)
            with open(quantized_path, "wb") as f:
                f.write(tflite_model_quantized)

            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)

            print(f" Quantized float16 TFLite model saved to: {quantized_path}")
            print(f"Original model size: {original_size:.2f} MB")
            print(f"Quantized float16 model size: {quantized_size:.2f} MB")
            reduction_abs = original_size - quantized_size
            reduction_perc = (
                (reduction_abs / original_size) * 100 if original_size > 0 else 0
            )
            print(f"Size reduction: {reduction_abs:.2f} MB ({reduction_perc:.2f}%)")

        except Exception as e:
            print(f"Error during TFLite float16 conversion: {e}")
            import traceback

            traceback.print_exc()

    def load_model(
        self,
        model_path: str = "model/hand_drawn_character_model.keras",
        label_map_path: str = "model/label_map.json",
        tflite_model_path: str = "model/hand_drawn_character_model_quant.tflite",
        use_quantize_model: bool = True,
    ) -> bool:
        """
        Loads a trained model and corresponding label map.

        Args:
            model_path (str, optional): Path to model file.
            label_map_path (str, optional): Path to label map JSON.
            tflite_model_path (str, optional): Path to TFLite model file.
            use_quantize_model (bool, optional): Whether to use quantized model. Defaults to True.

        Returns:
            bool: True if loading is successful, False otherwise.
        """
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)

            if use_quantize_model:
                self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                self.interpreter.allocate_tensors()

                # Get input and output tensor details for easier access during prediction
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                print("Successfully loaded TFLite model")
                return True
            else:
                self.model = keras.models.load_model(model_path)
                print("Successfully loaded Keras model")
                return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def predict(
        self, canvas: np.ndarray, use_quantize_model: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Predicts the character from a given image.

        Args:
            canvas (np.ndarray): Input image array (grayscale or BGR).
            use_quantize_model (bool, optional): Whether to use quantized model. Defaults to True.

        Returns:
            Tuple[Optional[str], float]: Predicted label and confidence score.
        """

        if self.label_map is None:
            print("Label map not loaded.")
            return None, 0.0

        # Ensure the canvas is in grayscale (single channel)
        if len(canvas.shape) == 3:  # If it's a 3-channel image (BGR)
            img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:  # If it's already grayscale
            img = canvas

        # Resize the image to match the model's input size
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # Normalize the image
        img = img / 255.0

        if use_quantize_model:
            # Prepare input for TFLite model (add batch and channel dimensions)
            img = np.expand_dims(img, axis=[0, -1]).astype(np.float32)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]["index"], img)

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            prediction = self.interpreter.get_tensor(self.output_details[0]["index"])
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
        else:
            # Prepare input for Keras model (add batch and channel dimensions)
            img = np.expand_dims(img, axis=[0, -1])

            # Make prediction
            prediction = self.model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

        # Map the predicted class to the corresponding label
        reverse_label_map = {
            idx: char_label for char_label, idx in self.label_map.items()
        }
        predicted_label = reverse_label_map.get(int(predicted_class))

        if predicted_label is None:
            return "Unknown", 0.0
        return predicted_label, float(confidence)

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
            dataset = dataset.shuffle(1800)  # Shuffle only if requested

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, true_labels, label_map
