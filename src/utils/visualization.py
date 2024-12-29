import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class DisplayManager:
    def __init__(self, webcam_height, last_saved_file=""):
        self.webcam_height = webcam_height
        self.last_saved_file = last_saved_file

        # Try multiple possible font locations
        possible_fonts = [
            "src\\assets\\fonts\\Phetsarath_OT.ttf",
            "C:\\Windows\\Fonts\\Phetsarath_OT.ttf",
            "\\usr\\share\\fonts\\truetype\\lao\\Phetsarath_OT.ttf",
            os.path.join(os.path.dirname(__file__), "fonts", "Phetsarath_OT.ttf"),
            # Add more potential font paths here
        ]

        self.font_path = None
        for font in possible_fonts:
            if os.path.exists(font):
                self.font_path = font
                break

        if self.font_path is None:
            print("Warning: No suitable font found. Text display may be limited.")

    def draw_interface(self, frame, predicted_label="", confidence=0.0):
        try:
            if not predicted_label or confidence <= 0.3:
                return

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

            # Get frame dimensions
            height, width = frame.shape[:2]
            y_position = height - 50  # 50 pixels from bottom

            # Convert to PIL Image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Use font if available, otherwise use default
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, 24)
                else:
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"Font error: {e}")
                font = ImageFont.load_default()

            # Draw text at bottom position
            draw.text(
                (10, y_position),
                f"Predicted: {predicted_label} - ({confidence:.2f})",
                font=font,
                fill=(0, 255, 0),
                stroke_width=0.5,  # Add stroke for bold effect
                stroke_fill=(0, 255, 0),  # Same color as fill for consistent look
            )

            # Convert back to OpenCV format
            frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in draw_interface: {e}")

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
