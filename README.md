🖐️ Real-Time Hand Gesture Recognition AI
An end-to-end Computer Vision pipeline that detects and classifies hand gestures in real-time. This project uses Google MediaPipe for landmark extraction and a custom TensorFlow/Keras Neural Network for classification.
📺 Demo IMAGE <img width="150" height="150" alt="113" src="https://github.com/user-attachments/assets/73600be0-6187-4f47-aefe-6130422c8ccd" />
🚀 Key Features
Landmark-Based Detection: Instead of processing raw pixels, the system tracks 21 specific 3D hand landmarks, making it robust against background noise.
High-Speed Inference: Optimized for real-time performance (30+ FPS) on standard CPUs.
Custom Dataset: Includes a complete pipeline for collecting unique hand data and training a personalized model.
Accuracy: Achieved 100% training accuracy and 93%+ real-time confidence.
🏗️ The Pipeline
Data Collection (collect_data.py): Captures image frames from the webcam and organizes them into gesture-specific directories.
Feature Extraction (utils/landmarks_extractor.py): Uses MediaPipe to convert images into a CSV file containing 63 spatial coordinates (x, y, z for 21 joints).
Neural Network Training (train_model.py): A Multi-Layer Perceptron (MLP) built with TensorFlow that learns to map coordinates to gesture labels.
Live Inference (main.py): The final application that overlays the predicted label and confidence score on the live video feed.
🛠️ Tech Stack
OpenCV: Video stream processing and UI overlays.
MediaPipe: Hand tracking and landmark localization.
TensorFlow/Keras: Deep Learning model architecture and training.
NumPy & Pandas: Data manipulation and numerical processing.
