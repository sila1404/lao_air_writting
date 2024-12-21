import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os


def is_index_finger_up(hand_landmarks):
    # Get y coordinates of index finger landmarks
    index_tip = hand_landmarks.landmark[8].y
    index_dip = hand_landmarks.landmark[7].y
    index_pip = hand_landmarks.landmark[6].y
    index_mcp = hand_landmarks.landmark[5].y

    # Get y coordinates of other finger landmarks
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y
    thumb_tip = hand_landmarks.landmark[4].y

    # Check if index is pointing and other fingers are closed
    index_up = index_tip < index_dip < index_pip < index_mcp
    other_down = all(
        [
            middle_tip > index_pip,
            ring_tip > index_pip,
            pinky_tip > index_pip,
            thumb_tip > index_pip,
        ]
    )

    return index_up and other_down


def generate_filename():
    # Create datasets directory if it doesn't exist
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("datasets", f"drawing_{timestamp}.jpg")


def main():
    # Initialize MediaPipe components
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Get webcam dimensions
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create canvas (128x128)
    canvas_size = (128, 128)
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    prev_point = None

    # Calculate drawing area in webcam frame (maintaining aspect ratio)
    aspect_ratio = 1.0  # square canvas
    if webcam_height < webcam_width:
        drawing_height = int(webcam_height * 0.45)  # 45% of height
        drawing_width = int(drawing_height * aspect_ratio)
    else:
        drawing_width = int(webcam_width * 0.4)  # 40% of width
        drawing_height = int(drawing_width * aspect_ratio)

    # Calculate drawing area position (shifted right and up)
    right_margin = int(webcam_width * 0.15)  # 15% margin from right
    top_margin = int(webcam_height * 0.2)  # 20% margin from top
    drawing_x = webcam_width - drawing_width - right_margin  # Position from right
    drawing_y = top_margin

    # Drawing settings
    drawing_color = (255, 255, 255)  # White color
    line_thickness = 4  # Increased line thickness

    # Variable to store last saved filename
    last_saved_file = ""

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break

        # Flip frame horizontally for more intuitive drawing
        frame = cv2.flip(frame, 1)

        # Draw canvas boundary on webcam frame
        cv2.rectangle(
            frame,
            (drawing_x, drawing_y),
            (drawing_x + drawing_width, drawing_y + drawing_height),
            (0, 0, 255),
            2,
        )

        # Add boundary label
        cv2.putText(
            frame,
            "Drawing Area",
            (drawing_x, drawing_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hands.process(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Get index finger coordinate in webcam space
                finger_x = int(hand_landmark.landmark[8].x * webcam_width)
                finger_y = int(hand_landmark.landmark[8].y * webcam_height)

                # Convert finger position to canvas space if within drawing area
                if (
                    drawing_x <= finger_x <= drawing_x + drawing_width
                    and drawing_y <= finger_y <= drawing_y + drawing_height
                ):
                    # Map finger position to canvas coordinates
                    canvas_x = int(
                        ((finger_x - drawing_x) / drawing_width) * canvas_size[0]
                    )
                    canvas_y = int(
                        ((finger_y - drawing_y) / drawing_height) * canvas_size[1]
                    )

                    # Only draw if index finger is pointing
                    if is_index_finger_up(hand_landmark):
                        if prev_point is not None:
                            cv2.line(
                                canvas,
                                prev_point,
                                (canvas_x, canvas_y),
                                drawing_color,
                                line_thickness,
                            )
                        prev_point = (canvas_x, canvas_y)

                        # Draw current point indicator on webcam frame
                        cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)
                    else:
                        prev_point = None
                else:
                    prev_point = None

                # Draw hand landmark on webcam frame
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                )
        else:
            prev_point = None

        # Add instructions and last saved filename to frame
        cv2.putText(
            frame,
            "Point index finger to draw",
            (10, webcam_height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press: 'c'-clear, 's'-save, 'q'-quit",
            (10, webcam_height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if last_saved_file:
            cv2.putText(
                frame,
                f"Last saved: {os.path.basename(last_saved_file)}",
                (10, webcam_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Display windows
        cv2.imshow("Hand Detection", frame)

        # Show canvas enlarged for better visibility (512x512)
        canvas_display = cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Drawing Preview (128x128)", canvas_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("c"):  # Clear canvas
            canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
            last_saved_file = ""
        elif key == ord("s"):  # Save canvas
            filename = generate_filename()
            cv2.imwrite(filename, canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])
            last_saved_file = filename
            print(f"Drawing saved as '{filename}'")
            # Clear canvas after saving
            canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
