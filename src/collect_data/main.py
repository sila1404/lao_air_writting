import cv2
from datetime import datetime
import os
from utils import HandTracker, DrawingCanvas, DrawingArea, DisplayManager


def generate_filename():
    # Create datasets directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("datasets", f"drawing_{timestamp}.jpg")


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize components
    hand_tracker = HandTracker()
    drawing_canvas = DrawingCanvas()
    drawing_area = DrawingArea(webcam_width, webcam_height)
    display_manager = DisplayManager(webcam_height)

    # Calculate drawing area in webcam frame (maintaining aspect ratio)
    drawing_area.setup_drawing_area()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break

        # Flip frame horizontally for more intuitive drawing
        frame = cv2.flip(frame, 1)

        # Draw canvas boundary on webcam frame
        display_manager.draw_canvas_boundary(frame, drawing_area)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hand_tracker.process_frame(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Get index finger coordinate in webcam space
                finger_x = int(hand_landmark.landmark[8].x * webcam_width)
                finger_y = int(hand_landmark.landmark[8].y * webcam_height)

                # Convert finger position to canvas space if within drawing area
                canvas_coord = drawing_area.get_canvas_coordinates(
                    finger_x, finger_y, drawing_canvas.canvas_size
                )

                # Only draw if index finger is pointing
                if canvas_coord and hand_tracker.is_index_finger_up(hand_landmark):
                    if drawing_canvas.prev_point is not None:
                        drawing_canvas.draw_line(
                            drawing_canvas.prev_point, canvas_coord
                        )
                    drawing_canvas.prev_point = canvas_coord

                    # Draw current point indicator on webcam frame
                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)
                else:
                    drawing_canvas.prev_point = None

                # Draw hand landmark on webcam frame
                hand_tracker.draw_landmarks(frame, hand_landmark)
        else:
            drawing_canvas.prev_point = None

        # Add instructions and last saved filename to frame
        display_manager.draw_collect_interface(frame)

        # Display windows
        cv2.imshow("Hand Detection", frame)

        # Show canvas enlarged for better visibility (512x512)
        canvas_display = cv2.resize(
            drawing_canvas.get_canvas(), (512, 512), interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Drawing Preview (128x128)", canvas_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("c"):  # Clear canvas
            drawing_canvas.clear()
            display_manager.last_saved_file = ""
        elif key == ord("s"):  # Save canvas
            filename = generate_filename()
            cv2.imwrite(
                filename, drawing_canvas.get_canvas(), [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            display_manager.last_saved_file = filename
            print(f"Drawing saved as '{filename}'")
            # Clear canvas after saving
            drawing_canvas.clear()

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
