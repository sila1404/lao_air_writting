import mediapipe as mp


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def is_index_finger_up(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8].y
        index_dip = hand_landmarks.landmark[7].y
        index_pip = hand_landmarks.landmark[6].y
        index_mcp = hand_landmarks.landmark[5].y

        middle_tip = hand_landmarks.landmark[12].y
        ring_tip = hand_landmarks.landmark[16].y
        pinky_tip = hand_landmarks.landmark[20].y

        index_up = index_tip < index_dip < index_pip < index_mcp
        other_down = all(
            [
                middle_tip > index_pip,
                ring_tip > index_pip,
                pinky_tip > index_pip,
            ]
        )

        return index_up and other_down

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
