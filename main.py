import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. Load the brain you trained
model_path = 'gesture_model.h5'

if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found! Please run train_model.py first.")
else:
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Labels matching your folders: 0_thumbs_Up, 1_Peace, 2_Palm
        class_names = ['Thumbs Up', 'Peace', 'Palm']

        # 3. Start Video Capture
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Flip and convert to RGB
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the "skeleton" on the screen
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract coordinates for prediction
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])
                    
                    # Predict the gesture
                    prediction = model.predict(np.array([landmarks]), verbose=0)
                    classID = np.argmax(prediction)
                    confidence = np.max(prediction)
                    gesture = class_names[classID]

                    # Display label if confidence is high
                    if confidence > 0.8:
                        cv2.putText(img, f'{gesture} ({int(confidence*100)}%)', (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Recognition", img)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        print("Tip: If you see 'signature not found', delete gesture_model.h5 and re-run train_model.py")