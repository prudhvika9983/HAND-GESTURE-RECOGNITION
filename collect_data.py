import cv2
import os
import time

# 1. Define the gestures you want to collect
# These must match your folder names exactly
gestures = ['0_thumbs_Up', '1_Peace', '2_Palm']
base_path = 'data'
num_samples = 50  # How many images to take per gesture

# Create folders if they don't exist
if not os.path.exists(base_path):
    os.makedirs(base_path)

cap = cv2.VideoCapture(0)

for gesture in gestures:
    gesture_path = os.path.join(base_path, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

    print(f"Prepare to collect data for: {gesture}")
    print("Get your hand ready! Starting in 5 seconds...")
    time.sleep(5)

    count = 0
    while count < num_samples:
        success, img = cap.read()
        if not success:
            break

        # Save the image
        img_name = os.path.join(gesture_path, f'{gesture}_{count}.jpg')
        cv2.imwrite(img_name, img)
        
        # Show progress on screen
        cv2.putText(img, f"Collecting {gesture}: {count}/{num_samples}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", img)
        
        count += 1
        cv2.waitKey(100) # Short delay between shots

    print(f"Finished collecting {gesture}!")

cap.release()
cv2.destroyAllWindows()