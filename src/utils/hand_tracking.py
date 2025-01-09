import mediapipe as mp


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def is_hand_open(self, hand_landmarks):
        # Get y-coordinates for all finger tips and their respective PIPs
        thumb_tip = hand_landmarks.landmark[4].y
        thumb_ip = hand_landmarks.landmark[3].y

        index_tip = hand_landmarks.landmark[8].y
        index_pip = hand_landmarks.landmark[6].y

        middle_tip = hand_landmarks.landmark[12].y
        middle_pip = hand_landmarks.landmark[10].y

        ring_tip = hand_landmarks.landmark[16].y
        ring_pip = hand_landmarks.landmark[14].y

        pinky_tip = hand_landmarks.landmark[20].y
        pinky_pip = hand_landmarks.landmark[18].y

        # Check if all fingers are extended (tip is above pip)
        all_fingers_up = all(
            [
                thumb_tip < thumb_ip,  # Thumb is up
                index_tip < index_pip,  # Index is up
                middle_tip < middle_pip,  # Middle is up
                ring_tip < ring_pip,  # Ring is up
                pinky_tip < pinky_pip,  # Pinky is up
            ]
        )

        return all_fingers_up

    def process_frame(self, frame):
        return self.hands.process(frame)

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
        )
