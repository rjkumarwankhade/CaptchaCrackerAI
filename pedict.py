import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(char_set)
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

max_length = 5  

model = load_model("captcha_model_improved.keras")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def predict_captcha(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 50))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 50, 100, 1)
    predictions = model.predict(img)
    predicted_text = "".join(index_to_char[np.argmax(prob)] for prob in predictions[0])
    return predicted_text

captcha_image = r"C:\Users\hp\OneDrive\Desktop\Python Software\Dataset For Testing\2b827.png"
result = predict_captcha(captcha_image)
print("Predicted CAPTCHA:", result)
