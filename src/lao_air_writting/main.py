import cv2
from utils import (
    HandTracker,
    DrawingArea,
    DrawingCanvas,
    CharacterRecognitionModel,
    DisplayManager,
)


def main():
    # Initialize components
    cap = cv2.VideoCapture(0)
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hand_tracker = HandTracker()
    drawing_canvas = DrawingCanvas()
    drawing_area = DrawingArea(webcam_width, webcam_height)
    recognizer = CharacterRecognitionModel()
    display_manager = DisplayManager(webcam_height)

    # Load the model
    if not recognizer.load_model():
        print("Failed to load model. Exiting...")
        return

    # Prediction settings
    prediction_cooldown = 0
    PREDICTION_INTERVAL = 15
    predicted_label = ""
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hand_tracker.process_frame(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                finger_x = int(hand_landmark.landmark[8].x * webcam_width)
                finger_y = int(hand_landmark.landmark[8].y * webcam_height)

                canvas_coord = drawing_area.get_canvas_coordinates(
                    finger_x, finger_y, drawing_canvas.canvas_size
                )

                if canvas_coord and hand_tracker.is_index_finger_up(hand_landmark):
                    if drawing_canvas.prev_point is not None:
                        drawing_canvas.draw_line(
                            drawing_canvas.prev_point, canvas_coord
                        )
                    drawing_canvas.prev_point = canvas_coord
                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)
                else:
                    drawing_canvas.prev_point = None

                hand_tracker.draw_landmarks(frame, hand_landmark)

        # Make prediction
        if prediction_cooldown <= 0:
            predicted_label, confidence = recognizer.predict(
                drawing_canvas.get_canvas()
            )
            prediction_cooldown = PREDICTION_INTERVAL
            # Debug print
            print(f"Prediction: {predicted_label}, Confidence: {confidence}")
        prediction_cooldown -= 1

        # Update display
        display_manager.draw_canvas_boundary(frame, drawing_area)
        display_manager.draw_interface(frame, predicted_label, confidence)

        # Show windows
        cv2.imshow("Hand Detection", frame)
        canvas_display = cv2.resize(
            drawing_canvas.get_canvas(), (512, 512), interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Drawing Preview (128x128)", canvas_display)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            drawing_canvas.clear()
            predicted_label = ""
            confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()