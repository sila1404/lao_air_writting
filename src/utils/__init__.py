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

    # Check if index is pointing and other fingers are closed
    index_up = index_tip < index_dip < index_pip < index_mcp
    other_down = all(
        [
            middle_tip > index_pip,
            ring_tip > index_pip,
            pinky_tip > index_pip,
        ]
    )

    return index_up and other_down