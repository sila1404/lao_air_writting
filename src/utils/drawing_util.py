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
