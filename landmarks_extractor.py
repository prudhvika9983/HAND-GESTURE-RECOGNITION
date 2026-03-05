import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data_dir = 'data'
data = []

# Valid image extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    print(f"Processing category: {category}")
    
    for img_name in os.listdir(category_path):
        # ONLY process actual image files
        if not img_name.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(category_path, img_name)
        image = cv2.imread(img_path)
        
        # Check if image was actually loaded to avoid the cv2.error
        if image is None:
            print(f"Skipping broken file: {img_name}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])
                row.append(category.split('_')[0])
                data.append(row)

# Save to the NEW file to avoid PermissionError
df = pd.DataFrame(data)
df.to_csv('./gesture_data.csv', index=False)
print("Success! 'gesture_data.csv' has been created with all landmarks.")