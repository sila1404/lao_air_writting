import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from utils import (
    HandTracker,
    DrawingArea,
    DrawingCanvas,
    CharacterRecognitionModel,
    DisplayManager,
)


class HandDrawingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Drawing Recognition App")

        # Initialize components
        self.setup_video_components()
        self.setup_gui_components()
        self.setup_recognition_components()

        # Add key bindings
        self.window.bind("<Key>", self.handle_keypress)

        # Start video thread
        self.is_running = True
        self.thread = threading.Thread(target=self.update_video)
        self.thread.daemon = True
        self.thread.start()

    def handle_keypress(self, event):
        # Handle key press events
        if event.char == "c":
            self.clear_text()
        elif event.char == "q":
            self.quit_app()
        elif event.char == "d":
            self.delete_last_char()

    def setup_video_components(self):
        self.cap = cv2.VideoCapture(0)
        self.webcam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.webcam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def setup_recognition_components(self):
        self.hand_tracker = HandTracker()
        self.drawing_canvas = DrawingCanvas()
        self.drawing_area = DrawingArea(self.webcam_width, self.webcam_height)
        self.recognizer = CharacterRecognitionModel()
        self.display_manager = DisplayManager(self.webcam_height)

        if not self.recognizer.load_model():
            print("Failed to load model")
            self.window.destroy()
            return

        # Prediction variables
        self.predicted_label = "None"
        self.confidence = 0.0
        self.is_drawing = False
        self.frames_since_last_draw = 0
        self.FRAMES_BEFORE_PREDICT = 30

    def setup_gui_components(self):
        # Create main frames
        self.video_frame = ttk.Frame(self.window)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.control_frame = ttk.Frame(self.window)
        self.control_frame.grid(row=1, column=0, padx=10, pady=5)

        # Create video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)

        # Create canvas preview label
        self.canvas_label = ttk.Label(self.video_frame)
        self.canvas_label.grid(row=0, column=1, padx=10)

        # Create text input
        self.text_var = tk.StringVar()
        self.text_input = ttk.Entry(
            self.control_frame,
            textvariable=self.text_var,
            width=50,
            font=("Impact", 20),
        )
        self.text_input.grid(row=0, column=0, padx=5)

        # Create buttons
        self.delete_button = ttk.Button(
            self.control_frame, text="Delete (d)", command=self.delete_last_char
        )
        self.delete_button.grid(row=0, column=2, padx=5)

        self.clear_button = ttk.Button(
            self.control_frame, text="Clear (c)", command=self.clear_text
        )
        self.clear_button.grid(row=0, column=1, padx=5)

        self.quit_button = ttk.Button(
            self.control_frame, text="Quit (q)", command=self.quit_app
        )
        self.quit_button.grid(row=0, column=2, padx=5)

    def update_video(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results = self.hand_tracker.process_frame(rgb_frame)

            current_frame_drew = False

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    finger_x = int(hand_landmark.landmark[8].x * self.webcam_width)
                    finger_y = int(hand_landmark.landmark[8].y * self.webcam_height)

                    canvas_coord = self.drawing_area.get_canvas_coordinates(
                        finger_x, finger_y, self.drawing_canvas.canvas_size
                    )

                    if canvas_coord and self.hand_tracker.is_index_finger_up(
                        hand_landmark
                    ):
                        if self.drawing_canvas.prev_point is not None:
                            self.drawing_canvas.draw_line(
                                self.drawing_canvas.prev_point, canvas_coord
                            )
                            current_frame_drew = True
                            self.is_drawing = True
                            self.frames_since_last_draw = 0
                        self.drawing_canvas.prev_point = canvas_coord
                        cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)
                    else:
                        self.drawing_canvas.prev_point = None

                    self.hand_tracker.draw_landmarks(frame, hand_landmark)

            # If we didn't draw anything this frame but were drawing before
            if not current_frame_drew and self.is_drawing:
                self.frames_since_last_draw += 1

                # If enough frames have passed without drawing, make prediction
                if self.frames_since_last_draw >= self.FRAMES_BEFORE_PREDICT:
                    predicted_label, confidence = self.recognizer.predict(
                        self.drawing_canvas.get_canvas()
                    )
                    if confidence > 0.3:  # Only update if confidence is high enough
                        # Update the prediction display
                        self.predicted_label = predicted_label
                        self.confidence = confidence

                        # Immediately update text input
                        current_text = self.text_var.get()
                        self.text_var.set(current_text + predicted_label)

                        # Clear canvas immediately after prediction
                        self.drawing_canvas.clear()
                        self.predicted_label = "None"
                        self.confidence = 0.0

                    # Reset drawing state
                    self.is_drawing = False
                    self.frames_since_last_draw = 0
                    self.drawing_canvas.prev_point = None  # Reset previous point

            # Update display
            self.display_manager.draw_canvas_boundary(frame, self.drawing_area)
            cv2.putText(
                frame,
                "Point index finger to draw",
                (10, self.webcam_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)

            # Update video label
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

            # Update canvas preview
            canvas_display = cv2.resize(
                self.drawing_canvas.get_canvas(),
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas_rgb = cv2.cvtColor(canvas_display, cv2.COLOR_BGR2RGB)
            canvas_pil = Image.fromarray(canvas_rgb)
            canvas_tk = ImageTk.PhotoImage(image=canvas_pil)

            # Update canvas label
            self.canvas_label.configure(image=canvas_tk)
            self.canvas_label.image = canvas_tk

    def delete_last_char(self):
        # Get current text
        current_text = self.text_var.get()
        if current_text:
            # Remove last character
            new_text = current_text[:-1]
            # Update text input
            self.text_var.set(new_text)

    def clear_text(self):
        self.text_var.set("")  # This will clear all text in the input field

    def quit_app(self):
        self.is_running = False
        self.cap.release()
        self.window.quit()

    def run(self):
        self.window.mainloop()


def main():
    root = tk.Tk()
    app = HandDrawingApp(root)
    app.run()


if __name__ == "__main__":
    main()
