import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os

data_file = 'gesture_data.csv' # Matching the new name

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found!")
else:
    df = pd.read_csv(data_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Input(shape=(42,)), 
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting training...")
    model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test))
    model.save('gesture_model.h5')
    print("Model saved as gesture_model.h5")