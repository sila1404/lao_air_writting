import tkinter as tk
from tkinter import ttk, Scale
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import os
from utils import HandTracker, DrawingCanvas, DrawingArea


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Drawing Application")

        # Initialize Lao characters
        self.lao_vowels = [
            "xະ",
            "xາ",
            "xິ",
            "xີ",
            "xຶ",
            "xື",
            "xຸ",
            "xູ",
            "ເx",
            "ໂx",
            "ໃx",
            "ໄx",
            "xຽ",
            "xໍ",
            "xັ",
            "xົ",
        ]
        self.lao_consonants = [
            "ກ",
            "ຂ",
            "ຄ",
            "ງ",
            "ຈ",
            "ສ",
            "ຊ",
            "ຍ",
            "ດ",
            "ຕ",
            "ຖ",
            "ທ",
            "ນ",
            "ບ",
            "ປ",
            "ຜ",
            "ຝ",
            "ພ",
            "ຟ",
            "ມ",
            "ຢ",
            "ລ",
            "ວ",
            "ຫ",
            "ອ",
            "ຮ",
        ]

        # Initialize character tracking
        self.current_vowel_index = 0
        self.current_consonant_index = 0

        # Default drawing area settings
        self.width_scale_value = tk.DoubleVar(value=0.4)  # 40% of width
        self.height_scale_value = tk.DoubleVar(value=0.45)  # 45% of height
        self.right_margin_scale_value = tk.DoubleVar(value=0.15)  # 15% margin
        self.top_margin_scale_value = tk.DoubleVar(value=0.2)  # 20% margin

        # Initialize save folder selection (before setup_gui)
        self.save_folder = tk.StringVar(value="vowels")

        # Character selection variable
        self.selected_char = tk.StringVar()

        # Initialize components
        self.setup_camera()
        self.setup_tracking_components()
        self.setup_gui()

        # Set up initial values after GUI is created
        self.save_folder.trace_add("write", self.on_folder_change)
        self.selected_char.trace_add("write", self.on_character_selected)

        # Initialize the combobox with vowels (default)
        self.update_character_list()
        self.update_current_character()

        # Explicitly update the character display after GUI setup
        self.update_current_character()

        # Start the update loop
        self.update()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.webcam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.webcam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def setup_tracking_components(self):
        self.hand_tracker = HandTracker()
        self.drawing_canvas = DrawingCanvas()
        self.drawing_area = DrawingArea(self.webcam_width, self.webcam_height)
        self.drawing_area.setup_drawing_area()

        # Status variables
        self.is_recording = True
        self.last_saved_file = ""
        self.show_settings = False

    def setup_gui(self):
        # Create main container with padding
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Create left and right frames with clear separation
        self.left_frame = ttk.Frame(self.main_container)
        self.left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)

        # Create a scrollable right frame
        self.right_frame_container = ttk.Frame(self.main_container)
        self.right_frame_container.pack(
            side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True
        )

        # Create canvas and scrollbar for right frame
        self.right_canvas = tk.Canvas(self.right_frame_container)
        self.right_scrollbar = ttk.Scrollbar(
            self.right_frame_container,
            orient="vertical",
            command=self.right_canvas.yview,
        )

        # Configure canvas
        self.right_frame = ttk.Frame(self.right_canvas)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        # Pack scrollbar and canvas
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create window in canvas
        self.canvas_frame_window = self.right_canvas.create_window(
            (0, 0),
            window=self.right_frame,
            anchor="nw",
            width=self.right_canvas.winfo_width(),
        )

        # Left Frame Components (Camera and Instructions)
        # Add a label frame for camera section
        self.camera_frame = ttk.LabelFrame(
            self.left_frame, text="Camera Feed", padding="5"
        )
        self.camera_frame.pack(fill=tk.BOTH)

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()

        # Instructions in a labeled frame
        self.instruction_frame = ttk.LabelFrame(
            self.left_frame, text="Instructions", padding="5"
        )
        self.instruction_frame.pack(fill=tk.X, pady=(10, 0))

        instructions = """  
        1. Point your index finger to draw  
        2. Open hand to stop drawing  
        3. Select folder (Vowels/Consonants)  
        4. Draw the shown character  
        5. Click 'Save Drawing' when done  
        """
        self.instruction_label = ttk.Label(
            self.instruction_frame, text=instructions, justify=tk.LEFT
        )
        self.instruction_label.pack(padx=5, pady=5)

        # Right Frame Components
        # Character Display Section
        self.character_frame = ttk.LabelFrame(
            self.right_frame, text="Current Character", padding="10"
        )
        self.character_frame.pack(fill=tk.X, pady=(0, 10))

        self.character_label = ttk.Label(
            self.character_frame,
            text="",
            font=("Phetsarath OT", 48),
            anchor="center",
        )
        self.character_label.pack()

        self.progress_label = ttk.Label(
            self.character_frame, text="", font=("Phetsarath OT", 10)
        )
        self.progress_label.pack()

        # Drawing Canvas Section
        self.canvas_frame = ttk.LabelFrame(
            self.right_frame, text="Drawing Area", padding="5"
        )
        self.canvas_frame.pack(fill=tk.BOTH)

        self.canvas_label = ttk.Label(self.canvas_frame)
        self.canvas_label.pack(padx=5, pady=5)

        # Controls Section
        self.controls_frame = ttk.LabelFrame(
            self.right_frame, text="Controls", padding="10"
        )
        self.controls_frame.pack(fill=tk.X, pady=(10, 0))

        # Folder Selection
        self.folder_frame = ttk.Frame(self.controls_frame)
        self.folder_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.folder_frame, text="Save to:").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Radiobutton(
            self.folder_frame, text="Vowels", value="vowels", variable=self.save_folder
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            self.folder_frame,
            text="Consonants",
            value="consonants",
            variable=self.save_folder,
        ).pack(side=tk.LEFT, padx=5)

        # Character Selection
        self.char_selection_frame = ttk.Frame(self.controls_frame)
        self.char_selection_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            self.char_selection_frame,
            text="Start from:",
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.char_combobox = ttk.Combobox(
            self.char_selection_frame,
            textvariable=self.selected_char,
            state="readonly",
            font=("Phetsarath OT", 15),
            width=10,
        )
        self.char_combobox.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.button_frame = ttk.Frame(self.controls_frame)
        self.button_frame.pack(fill=tk.X)

        self.clear_btn = ttk.Button(
            self.button_frame, text="Clear Canvas", command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            self.button_frame, text="Save Drawing", command=self.save_drawing
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.settings_btn = ttk.Button(
            self.button_frame, text="Settings", command=self.toggle_settings
        )
        self.settings_btn.pack(side=tk.LEFT, padx=5)

        # Status Bar
        self.status_frame = ttk.Frame(self.right_frame)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(
            self.status_frame, text="Ready", relief=tk.SUNKEN, padding=(5, 2)
        )
        self.status_label.pack(fill=tk.X)

        # Settings Panel (Initially Hidden)
        self.settings_container = ttk.Frame(self.right_frame)
        self.setup_settings_panel()

        # Configure scrolling
        self.right_frame.bind("<Configure>", self.on_frame_configure)
        self.right_canvas.bind("<Configure>", self.on_canvas_configure)
        self.right_canvas.bind("<Enter>", self.bind_mousewheel)
        self.right_canvas.bind("<Leave>", self.unbind_mousewheel)

        # Bind mouse wheel to scroll
        self.right_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def setup_settings_panel(self):
        """Setup the settings panel with its own method for better organization"""
        # Create the settings frame
        self.settings_frame = ttk.LabelFrame(
            self.right_frame,  # Changed from settings_canvas to right_frame
            text="Drawing Area Settings",
            padding="10",
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

    def add_settings_controls(self):
        """Add controls to the settings panel"""
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

    def update_character_list(self):
        """Update the character list in the combobox based on selected folder"""
        if self.save_folder.get() == "vowels":
            char_list = self.lao_vowels
            current_index = self.current_vowel_index
        else:
            char_list = self.lao_consonants
            current_index = self.current_consonant_index

        # Update combobox values
        self.char_combobox["values"] = char_list

        # Set current character as selected
        if char_list:  # Make sure there are values
            self.char_combobox.set(char_list[current_index])

    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match"""
        width = event.width
        self.right_canvas.itemconfig(self.canvas_frame_window, width=width)

    def bind_mousewheel(self, event):
        """Bind mousewheel to scroll only when cursor is over the canvas"""
        self.right_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def unbind_mousewheel(self, event):
        """Unbind mousewheel when cursor leaves the canvas"""
        self.right_canvas.unbind_all("<MouseWheel>")

    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def toggle_settings(self):
        """Toggle the visibility of settings panel"""
        if hasattr(self, "show_settings") and self.show_settings:
            self.settings_frame.pack_forget()
            self.show_settings = False
            self.settings_btn.configure(text="Show Settings")
        else:
            self.settings_frame.pack(before=self.status_frame, pady=10, fill=tk.X)
            self.show_settings = True
            self.settings_btn.configure(text="Hide Settings")

            # Update scroll region
            self.right_frame.update_idletasks()
            self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def update_drawing_area(self, *args):
        # Update DrawingArea parameters
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

        # Update drawing area position
        self.drawing_area.drawing_x = (
            self.webcam_width
            - self.drawing_area.drawing_width
            - self.drawing_area.right_margin
        )
        self.drawing_area.drawing_y = self.drawing_area.top_margin

    def update(self):
        if self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                # Draw canvas boundary
                cv2.rectangle(
                    frame,
                    (self.drawing_area.drawing_x, self.drawing_area.drawing_y),
                    (
                        self.drawing_area.drawing_x + self.drawing_area.drawing_width,
                        self.drawing_area.drawing_y + self.drawing_area.drawing_height,
                    ),
                    (0, 255, 0),
                    2,
                )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hand_tracker.process_frame(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        finger_x = int(hand_landmark.landmark[8].x * self.webcam_width)
                        finger_y = int(hand_landmark.landmark[8].y * self.webcam_height)

                        # Convert to canvas space
                        canvas_coord = self.drawing_area.get_canvas_coordinates(
                            finger_x, finger_y, self.drawing_canvas.canvas_size
                        )

                        if canvas_coord:
                            # Update current position on canvas
                            self.drawing_canvas.update_tracking_position(canvas_coord)

                            # Draw if index finger is pointing
                            if not self.hand_tracker.is_hand_open(hand_landmark):
                                if self.drawing_canvas.prev_point is not None:
                                    self.drawing_canvas.draw_line(
                                        self.drawing_canvas.prev_point, canvas_coord
                                    )
                                self.drawing_canvas.prev_point = canvas_coord
                                self.drawing_canvas.update_drawing_position(
                                    canvas_coord
                                )
                            else:
                                self.drawing_canvas.prev_point = None

                        # Draw hand landmarks
                        self.hand_tracker.draw_landmarks(frame, hand_landmark)
                else:
                    self.drawing_canvas.prev_point = None
                    self.drawing_canvas.current_pos = (
                        None  # Clear current position when hand is not detected
                    )

                # Update camera feed
                camera_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                camera_image = ImageTk.PhotoImage(camera_image)
                self.camera_label.configure(image=camera_image)
                self.camera_label.image = camera_image

                # Update canvas preview
                canvas_display = cv2.resize(
                    self.drawing_canvas.get_canvas(),
                    self.drawing_canvas.canvas_size,
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas_image = Image.fromarray(canvas_display)
                canvas_image = ImageTk.PhotoImage(canvas_image)
                self.canvas_label.configure(image=canvas_image)
                self.canvas_label.image = canvas_image

        # Schedule next update
        self.root.after(10, self.update)

    def clear_canvas(self):
        self.drawing_canvas.clear()
        self.last_saved_file = ""
        self.status_label.config(text="Canvas cleared")

    def update_current_character(self):
        """Update the display of the current character and progress"""
        if self.save_folder.get() == "vowels":
            current_char = self.lao_vowels[self.current_vowel_index]
            progress = f"Vowel {self.current_vowel_index + 1}/{len(self.lao_vowels)}"
        else:
            current_char = self.lao_consonants[self.current_consonant_index]
            progress = f"Consonant {self.current_consonant_index + 1}/{len(self.lao_consonants)}"

        self.character_label.config(text=current_char)
        self.progress_label.config(text=progress)

        # Update combobox selection to match current character
        self.char_combobox.set(current_char)

    def on_folder_change(self, *args):
        """Handle folder selection change"""
        self.update_character_list()
        self.update_current_character()

    def on_character_selected(self, *args):
        """Handle character selection from combobox"""
        selected = self.selected_char.get()
        if selected:
            if self.save_folder.get() == "vowels":
                self.current_vowel_index = self.lao_vowels.index(selected)
            else:
                self.current_consonant_index = self.lao_consonants.index(selected)
            self.update_current_character()

    def next_character(self):
        """Move to the next character in the current category"""
        if self.save_folder.get() == "vowels":
            self.current_vowel_index = (self.current_vowel_index + 1) % len(
                self.lao_vowels
            )
        else:
            self.current_consonant_index = (self.current_consonant_index + 1) % len(
                self.lao_consonants
            )
        self.update_current_character()

    def save_drawing(self):
        try:
            if self.drawing_canvas.canvas.max() == 0:
                self.status_label.config(text="Cannot save empty drawing!")
                print("Drawing is empty - nothing to save")
                return

            # Create main datasets directory if it doesn't exist
            if not os.path.exists("datasets"):
                os.makedirs("datasets")

            # Create vowels/consonants directories if they don't exist
            vowels_dir = os.path.join("datasets", "vowels")
            consonants_dir = os.path.join("datasets", "consonants")

            if not os.path.exists(vowels_dir):
                os.makedirs(vowels_dir)
            if not os.path.exists(consonants_dir):
                os.makedirs(consonants_dir)

            # Get current character
            current_char = (
                self.lao_vowels[self.current_vowel_index]
                if self.save_folder.get() == "vowels"
                else self.lao_consonants[self.current_consonant_index]
            )

            # Create character-specific directory
            char_dir = os.path.join(
                "datasets", self.save_folder.get(), f"char_{ord(current_char):04d}"
            )

            if not os.path.exists(char_dir):
                os.makedirs(char_dir)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(char_dir, f"{timestamp}.jpg")

            # Save only the drawing canvas without the overlay
            success = cv2.imwrite(
                filename, self.drawing_canvas.canvas, [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

            if success:
                print(f"Successfully saved to {filename}")
                self.last_saved_file = filename

                # Create a metadata file to store character information
                metadata_file = os.path.join(char_dir, "metadata.txt")
                if not os.path.exists(metadata_file):
                    with open(metadata_file, "w", encoding="utf-8") as f:
                        f.write(f"Character: {current_char}\n")
                        f.write(f"Unicode: {ord(current_char)}\n")

                self.status_label.config(
                    text=f"Saved '{current_char}' as '{os.path.basename(filename)}' in char_{ord(current_char):04d}"
                )
                self.drawing_canvas.clear()
                self.next_character()
            else:
                print("Failed to save image - cv2.imwrite returned False")
                self.status_label.config(text="Failed to save image!")

        except Exception as e:
            self.status_label.config(text=f"Error saving file: {str(e)}")

    def on_closing(self):
        self.is_recording = False
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
