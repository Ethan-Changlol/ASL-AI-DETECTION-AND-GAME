import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
clf = joblib.load("asl.joblib")  # Make sure your model is trained and saved

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Convert landmarks into normalized [x,y,z] list"""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    zs = [lm.z for lm in hand_landmarks.landmark]

    base_x, base_y, base_z = xs[0], ys[0], zs[0]  # wrist as reference

    landmarks = []
    for x, y, z in zip(xs, ys, zs):
        landmarks.append(x - base_x)
        landmarks.append(y - base_y)
        landmarks.append(z - base_z)

    return landmarks

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        predicted_letter = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                try:
                    landmarks = extract_landmarks(hand_landmarks)
                    predicted_letter = clf.predict([landmarks])[0]
                except:
                    predicted_letter = "?"

        # Display predicted sign
        cv2.putText(image, f"Sign: {predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("ASL Sign Detection", image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
