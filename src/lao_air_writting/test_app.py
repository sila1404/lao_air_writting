import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
from PIL import Image, ImageTk
import cv2
from utils import (
    HandTracker,
    DrawingCanvas,
    DrawingArea,
    CharacterRecognitionModel,
    OCRProcessor,
)


class TestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lao Character Recognition Testing")

        # Initialize models
        self.char_recognition = CharacterRecognitionModel()
        self.char_recognition.load_model()
        self.ocr_processor = OCRProcessor()

        # Configure style
        style = ttk.Style()
        style.configure("TLabelframe", padding=5)
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=2)

        # Initialize main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Create navigation frame
        self.nav_frame = ttk.Frame(self.main_container)
        self.nav_frame.pack(fill=tk.X, pady=(0, 10))

        # Navigation buttons
        self.hand_btn = ttk.Button(
            self.nav_frame,
            text="Character Recognition Test",
            command=lambda: self.show_page("hand"),
        )
        self.hand_btn.pack(side=tk.LEFT, padx=5)

        self.ocr_btn = ttk.Button(
            self.nav_frame, text="OCR Test", command=lambda: self.show_page("ocr")
        )
        self.ocr_btn.pack(side=tk.LEFT, padx=5)

        # Create frames for both pages
        self.hand_page = HandWritingPage(self.main_container, self.char_recognition)
        self.ocr_page = OCRPage(self.main_container, self.ocr_processor)

        # Show hand writing page by default
        self.current_page = "hand"
        self.show_page("hand")

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_page(self, page):
        # Hide both pages
        self.hand_page.frame.pack_forget()
        self.ocr_page.frame.pack_forget()

        # Show selected page
        if page == "hand":
            self.hand_page.frame.pack(fill=tk.BOTH, expand=True)
            self.hand_btn.state(["disabled"])
            self.ocr_btn.state(["!disabled"])
        else:
            self.ocr_page.frame.pack(fill=tk.BOTH, expand=True)
            self.ocr_btn.state(["disabled"])
            self.hand_btn.state(["!disabled"])

        self.current_page = page

    def on_closing(self):
        if hasattr(self.hand_page, "on_closing"):
            self.hand_page.on_closing()
        self.root.destroy()


class HandWritingPage:
    def __init__(self, parent, model):
        self.frame = ttk.Frame(parent)
        self.parent = parent
        self.model = model

        # Initialize variables for drawing area settings
        self.width_scale_value = tk.DoubleVar(value=0.4)
        self.height_scale_value = tk.DoubleVar(value=0.4)
        self.right_margin_scale_value = tk.DoubleVar(value=0.15)
        self.top_margin_scale_value = tk.DoubleVar(value=0.2)
        self.show_settings = False

        # Create scrollable right frame
        self.right_canvas = tk.Canvas(self.frame)
        self.right_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.right_scrollbar = ttk.Scrollbar(
            self.frame, orient="vertical", command=self.right_canvas.yview
        )
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.right_frame = ttk.Frame(self.right_canvas)
        self.right_canvas.create_window((0, 0), window=self.right_frame, anchor="nw")

        # Configure scrolling
        self.right_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(
                scrollregion=self.right_canvas.bbox("all")
            ),
        )
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        # Left frame for camera
        self.left_frame = ttk.Frame(self.frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)

        # Camera section
        self.camera_frame = ttk.LabelFrame(
            self.left_frame, text="Camera Feed", padding="5"
        )
        self.camera_frame.pack(fill=tk.BOTH)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()

        # Instructions
        self.instruction_frame = ttk.LabelFrame(
            self.left_frame, text="Instructions", padding="5"
        )
        self.instruction_frame.pack(fill=tk.X, pady=(10, 0))

        instructions = """  
        1. Point your index finger to draw  
        2. Open hand to stop drawing  
        3. Draw any Lao character  
        4. Click 'Recognize' or press 'R' to test  
        5. Click 'Clear' or press 'C' to clear  
        """
        self.instruction_label = ttk.Label(
            self.instruction_frame, text=instructions, justify=tk.LEFT
        )
        self.instruction_label.pack(padx=5, pady=5)

        # Drawing Canvas Section
        self.canvas_frame = ttk.LabelFrame(
            self.right_frame, text="Drawing Area", padding="5"
        )
        self.canvas_frame.pack(fill=tk.BOTH, pady=5)

        self.canvas_container = ttk.Frame(self.canvas_frame)
        self.canvas_container.pack(pady=5)

        self.canvas_size = 400
        self.canvas_label = ttk.Label(self.canvas_container)
        self.canvas_label.pack()

        # Result section
        self.result_frame = ttk.LabelFrame(
            self.right_frame, text="Recognition Result", padding="10"
        )
        self.result_frame.pack(fill=tk.X, pady=(10, 0))

        self.result_label = ttk.Label(
            self.result_frame,
            text="Draw a character and click Recognize",
            font=("Saysettha OT", 16),
        )
        self.result_label.pack()

        # Control buttons
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.pack(pady=10)

        self.clear_btn = ttk.Button(
            self.button_frame, text="Clear Canvas", command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.recognize_btn = ttk.Button(
            self.button_frame, text="Recognize", command=self.recognize_character
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        # Settings Button
        self.settings_btn = ttk.Button(
            self.right_frame, text="Show Settings", command=self.toggle_settings
        )
        self.settings_btn.pack(pady=5)

        # Settings Panel (initially hidden)
        self.setup_settings_panel()

        # Initialize components
        self.setup_camera()
        self.setup_tracking_components()

        # Bind keyboard shortcuts
        self.parent.bind("<c>", lambda e: self.clear_canvas())
        self.parent.bind("<r>", lambda e: self.recognize_character())

        # Bind mousewheel
        self.right_canvas.bind("<Enter>", self.bind_mousewheel)
        self.right_canvas.bind("<Leave>", self.unbind_mousewheel)

        # Start update loop
        self.update()

    def setup_settings_panel(self):
        """Setup the settings panel with its own method for better organization"""
        self.settings_frame = ttk.LabelFrame(
            self.right_frame, text="Drawing Area Settings", padding="10"
        )

        # Add settings controls with better spacing and organization
        # Width scale
        ttk.Label(self.settings_frame, text="Width (% of screen)").pack(pady=(0, 2))
        Scale(
            self.settings_frame,
            from_=0.1,
            to=0.8,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.width_scale_value,
            command=self.update_drawing_area,
        ).pack(fill=tk.X, padx=5)

        # Height scale
        ttk.Label(self.settings_frame, text="Height (% of screen)").pack(pady=(10, 2))
        Scale(
            self.settings_frame,
            from_=0.1,
            to=0.8,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.height_scale_value,
            command=self.update_drawing_area,
        ).pack(fill=tk.X, padx=5)

        # Right margin scale
        ttk.Label(self.settings_frame, text="Right margin (% of screen)").pack(
            pady=(10, 2)
        )
        Scale(
            self.settings_frame,
            from_=0,
            to=0.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.right_margin_scale_value,
            command=self.update_drawing_area,
        ).pack(fill=tk.X, padx=5)

        # Top margin scale
        ttk.Label(self.settings_frame, text="Top margin (% of screen)").pack(
            pady=(10, 2)
        )
        Scale(
            self.settings_frame,
            from_=0,
            to=0.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.top_margin_scale_value,
            command=self.update_drawing_area,
        ).pack(fill=tk.X, padx=5)

    def toggle_settings(self):
        """Toggle the visibility of settings panel"""
        if self.show_settings:
            self.settings_frame.pack_forget()
            self.show_settings = False
            self.settings_btn.configure(text="Show Settings")
        else:
            self.settings_frame.pack(after=self.settings_btn, pady=10, fill=tk.X)
            self.show_settings = True
            self.settings_btn.configure(text="Hide Settings")

        # Update scroll region
        self.right_frame.update_idletasks()
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def bind_mousewheel(self, event):
        """Bind mousewheel to scroll only when cursor is over the canvas"""
        self.right_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def unbind_mousewheel(self, event):
        """Unbind mousewheel when cursor leaves the canvas"""
        self.right_canvas.unbind_all("<MouseWheel>")

    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_drawing_area(self, *args):
        """Update drawing area dimensions based on scale values"""
        if hasattr(self, "drawing_area"):
            self.drawing_area.drawing_width = int(
                self.webcam_width * self.width_scale_value.get()
            )
            self.drawing_area.drawing_height = int(
                self.webcam_height * self.height_scale_value.get()
            )
            self.drawing_area.right_margin = int(
                self.webcam_width * self.right_margin_scale_value.get()
            )
            self.drawing_area.top_margin = int(
                self.webcam_height * self.top_margin_scale_value.get()
            )
            self.drawing_area.drawing_x = (
                self.webcam_width
                - self.drawing_area.drawing_width
                - self.drawing_area.right_margin
            )
            self.drawing_area.drawing_y = self.drawing_area.top_margin

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.webcam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.webcam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_recording = True

    def setup_tracking_components(self):
        self.hand_tracker = HandTracker()
        self.drawing_canvas = DrawingCanvas()
        self.drawing_area = DrawingArea(self.webcam_width, self.webcam_height)

    def update(self):
        if self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Draw the drawing area boundary on camera feed
                cv2.rectangle(
                    frame,
                    (self.drawing_area.drawing_x, self.drawing_area.drawing_y),
                    (
                        self.drawing_area.drawing_x + self.drawing_area.drawing_width,
                        self.drawing_area.drawing_y + self.drawing_area.drawing_height,
                    ),
                    (0, 255, 0),  # Green color
                    2,  # Thickness
                )

                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hand_tracker.process_frame(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        # Get index finger tip coordinates
                        finger_x = int(hand_landmark.landmark[8].x * self.webcam_width)
                        finger_y = int(hand_landmark.landmark[8].y * self.webcam_height)

                        # Get canvas coordinates
                        canvas_coord = self.drawing_area.get_canvas_coordinates(
                            finger_x, finger_y, self.drawing_canvas.canvas_size
                        )

                        if canvas_coord:
                            self.drawing_canvas.update_tracking_position(canvas_coord)

                            if not self.hand_tracker.is_hand_open(hand_landmark):
                                if self.drawing_canvas.prev_point is not None:
                                    self.drawing_canvas.draw_line(
                                        self.drawing_canvas.prev_point, canvas_coord
                                    )
                                self.drawing_canvas.prev_point = canvas_coord
                            else:
                                self.drawing_canvas.prev_point = None

                        self.hand_tracker.draw_landmarks(frame, hand_landmark)

                # Update displays
                camera_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                camera_image = ImageTk.PhotoImage(camera_image)
                self.camera_label.configure(image=camera_image)
                self.camera_label.image = camera_image

                canvas_display = cv2.resize(
                    self.drawing_canvas.get_canvas(),
                    (256, 256),
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas_image = Image.fromarray(canvas_display)
                canvas_image = ImageTk.PhotoImage(canvas_image)
                self.canvas_label.configure(image=canvas_image)
                self.canvas_label.image = canvas_image

        self.frame.after(10, self.update)

    def clear_canvas(self):
        self.drawing_canvas.clear()
        self.result_label.config(
            text="Draw a character and click Recognize", font=("Saysettha OT", 16)
        )

    def recognize_character(self):
        try:
            canvas_img = self.drawing_canvas.get_canvas()
            predicted_char, confidence = self.model.predict(canvas_img)

            self.result_label.config(
                text=f"Predicted: {predicted_char}\nConfidence: {(confidence * 100):.2f}%",
                font=("Saysettha OT", 16),
            )
        except Exception as e:
            self.result_label.config(
                text=f"Error during recognition: {str(e)}", font=("Saysettha OT", 16)
            )

    def on_closing(self):
        self.is_recording = False
        if hasattr(self, "cap"):
            self.cap.release()


class OCRPage:
    def __init__(self, parent, ocr_processor):
        self.frame = ttk.Frame(parent)
        self.parent = parent
        self.ocr_processor = ocr_processor

        # Store current image
        self.current_image = None
        self.current_image_path = None

        # Image display section
        self.image_frame = ttk.LabelFrame(
            self.frame, text="Image Preview", padding="10"
        )
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True)

        # Controls
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        self.select_btn = ttk.Button(
            self.control_frame, text="Select Image", command=self.select_image
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(
            self.control_frame, text="Process OCR", command=self.process_ocr
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Results
        self.result_frame = ttk.LabelFrame(self.frame, text="OCR Results", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            self.result_frame, height=10, font=("Saysettha OT", 12), wrap=tk.WORD
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(
            self.result_frame, orient="vertical", command=self.result_text.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=self.scrollbar.set)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                self.current_image_path = file_path
                image = Image.open(file_path)

                # Resize for display
                display_size = (800, 600)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)

                self.current_image = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.current_image)

                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(
                    tk.END, "Image loaded. Click 'Process OCR' to begin recognition."
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def process_ocr(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        try:
            image = cv2.imread(self.current_image_path)
            detected_text = self.ocr_processor.recognize_text(image)

            self.result_text.delete(1.0, tk.END)
            if detected_text:
                self.result_text.insert(tk.END, "Detected Text:\n\n")
                self.result_text.insert(tk.END, detected_text)
            else:
                self.result_text.insert(tk.END, "No text detected in the image.")

        except Exception as e:
            messagebox.showerror("Error", f"OCR processing failed: {str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error during OCR processing: {str(e)}")


def main():
    root = tk.Tk()
    app = TestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
