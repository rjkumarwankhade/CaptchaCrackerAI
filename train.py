import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Reshape, Activation, Dropout)
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Define Character Set and Parameters
# -------------------------------
char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(char_set)
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

max_length = 5  

# -------------------------------
# 2. Data Loading and Preprocessing
# -------------------------------
def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(data_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 50))
            X.append(img)
            y.append(os.path.splitext(file)[0])
    return np.array(X), np.array(y)

def encode_labels(labels, max_length):
    encoded = np.zeros((len(labels), max_length, num_classes))
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            encoded[i, j, char_to_index[char]] = 1
    return encoded

data_dir = rf"C:\Users\hp\OneDrive\Desktop\Python Software\Dataset For Training"  
X, y = load_data(data_dir)

X = X.astype(np.float32) / 255.0
X = X.reshape(-1, 50, 100, 1)
y_encoded = encode_labels(y, max_length)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -------------------------------
# 3. Build and Train the Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(50, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(max_length * num_classes, activation="linear"),
    Reshape((max_length, num_classes)),
    Activation("softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=40, validation_data=(X_test, y_test))
model.save("captcha_model_improved.keras")
print("Training completed and model saved.")
