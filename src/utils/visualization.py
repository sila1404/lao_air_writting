import cv2
import os


class DisplayManager:
    def __init__(self, webcam_height, last_saved_file=""):
        self.webcam_height = webcam_height
        self.last_saved_file = last_saved_file

    def draw_interface(self, frame, predicted_label="", confidence=0.0):
        cv2.putText(
            frame,
            "Point index finger to draw",
            (10, self.webcam_height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press: 'c'-clear, 'q'-quit",
            (10, self.webcam_height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if predicted_label:
            cv2.putText(
                frame,
                f"Prediction: {predicted_label} ({confidence:.2f})",
                (10, self.webcam_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def draw_canvas_boundary(self, frame, drawing_area):
        cv2.rectangle(
            frame,
            (drawing_area.drawing_x, drawing_area.drawing_y),
            (
                drawing_area.drawing_x + drawing_area.drawing_width,
                drawing_area.drawing_y + drawing_area.drawing_height,
            ),
            (0, 0, 255),
            2,
        )

        # Add boundary label
        cv2.putText(
            frame,
            "Drawing Area",
            (drawing_area.drawing_x, drawing_area.drawing_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    def draw_collect_interface(self, frame):
        cv2.putText(
            frame,
            "Point index finger to draw",
            (10, self.webcam_height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press: 'c'-clear, 's'-save, 'q'-quit",
            (10, self.webcam_height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if self.last_saved_file:
            cv2.putText(
                frame,
                f"Last saved: {os.path.basename(self.last_saved_file)}",
                (10, self.webcam_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def plot_training_history(self, history):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()
