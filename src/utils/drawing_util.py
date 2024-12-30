import cv2
import numpy as np


class DrawingCanvas:
    def __init__(self, canvas_size=(128, 128)):
        self.canvas_size = canvas_size
        self.canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing_color = (255, 255, 255)
        self.line_thickness = 3

    def clear(self):
        self.canvas = np.zeros(
            (self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8
        )
        self.prev_point = None

    def draw_line(self, start_point, end_point):
        if start_point is not None and end_point is not None:
            cv2.line(
                self.canvas,
                start_point,
                end_point,
                self.drawing_color,
                self.line_thickness,
            )

    def get_canvas(self):
        return self.canvas


class DrawingArea:
    def __init__(self, webcam_width, webcam_height):
        self.webcam_width = webcam_width
        self.webcam_height = webcam_height
        self.setup_drawing_area()

    def setup_drawing_area(self):
        aspect_ratio = 1.0  # square canvas
        if self.webcam_height < self.webcam_width:
            self.drawing_height = int(self.webcam_height * 0.45)  # 45% of height
            self.drawing_width = int(self.drawing_height * aspect_ratio)
        else:
            self.drawing_width = int(self.webcam_width * 0.4)  # 40% of width
            self.drawing_height = int(self.drawing_width * aspect_ratio)

        # Calculate drawing area position (shifted right and up)
        self.right_margin = int(self.webcam_width * 0.15)  # 15% margin from right
        self.top_margin = int(self.webcam_height * 0.2)  # 20% margin from top
        self.drawing_x = (
            self.webcam_width - self.drawing_width - self.right_margin
        )  # Position from right
        self.drawing_y = self.top_margin

    def get_canvas_coordinates(self, finger_x, finger_y, canvas_size):
        if (
            self.drawing_x <= finger_x <= self.drawing_x + self.drawing_width
            and self.drawing_y <= finger_y <= self.drawing_y + self.drawing_height
        ):
            canvas_x = int(
                ((finger_x - self.drawing_x) / self.drawing_width) * canvas_size[0]
            )
            canvas_y = int(
                ((finger_y - self.drawing_y) / self.drawing_height) * canvas_size[1]
            )
            return (canvas_x, canvas_y)
        return None


class DrawingCenterer:
    def __init__(self, canvas_size=(128, 128)):  # Changed to 128x128
        self.canvas_size = canvas_size

    def center_drawing(self, canvas):
        # Create a copy of the canvas to work with
        canvas_copy = canvas.copy()

        # Find non-zero points (where drawing exists)
        y_coords, x_coords = np.nonzero(cv2.cvtColor(canvas_copy, cv2.COLOR_BGR2GRAY))

        if len(x_coords) == 0 or len(y_coords) == 0:  # If no drawing
            return cv2.resize(
                canvas_copy, self.canvas_size
            )  # Ensure correct output size

        # Get bounding box
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        # Get drawing dimensions
        drawing_width = max_x - min_x + 1
        drawing_height = max_y - min_y + 1

        if drawing_width <= 1 or drawing_height <= 1:
            return cv2.resize(canvas_copy, self.canvas_size)

        # Crop the drawing
        cropped = canvas_copy[min_y : max_y + 1, min_x : max_x + 1]

        # Calculate scaling to fit in center while maintaining aspect ratio
        scale = min(
            (self.canvas_size[0] * 0.6) / drawing_width,
            (self.canvas_size[1] * 0.6) / drawing_height,
        )

        # Calculate new dimensions
        new_width = max(int(drawing_width * scale), 1)
        new_height = max(int(drawing_height * scale), 1)

        # Create a new blank canvas with the correct size
        centered_canvas = np.zeros(
            (self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8
        )

        try:
            # Resize the cropped drawing
            resized = cv2.resize(cropped, (new_width, new_height))

            # Calculate position to paste the resized drawing
            x_offset = (self.canvas_size[0] - new_width) // 2
            y_offset = (self.canvas_size[1] - new_height) // 2

            # Ensure the offsets and dimensions are within bounds
            x_offset = max(0, min(x_offset, self.canvas_size[0] - new_width))
            y_offset = max(0, min(y_offset, self.canvas_size[1] - new_height))

            # Paste the resized drawing in the center
            centered_canvas[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = resized

        except Exception as e:
            print(f"Error during centering: {e}")
            return cv2.resize(canvas_copy, self.canvas_size)

        return centered_canvas
