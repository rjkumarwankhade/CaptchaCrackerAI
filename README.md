# CaptchaCrackerAI
DeepCaptchaSolver is a CNN-based CAPTCHA solver that recognizes alphanumeric characters (A-Z, a-z, 0-9) from 100x50 grayscale images. Built with TensorFlow, OpenCV, and NumPy, it supports training, saving, and real-time predictions using one-hot encoding.



## Overview
This project implements a CAPTCHA recognition system using a Convolutional Neural Network (CNN) in TensorFlow/Keras. It is designed to recognize alphanumeric CAPTCHA images (A-Z, a-z, 0-9) with a fixed length of 5 characters. The model is trained on grayscale CAPTCHA images and can predict text from new CAPTCHA images.

## Features
- Preprocessing of CAPTCHA images (grayscale conversion, resizing, normalization)
- One-hot encoding of CAPTCHA labels
- CNN-based deep learning model for multi-character recognition
- Training and validation split for effective learning
- Model saving and loading for future predictions
- CAPTCHA prediction function to decode test images

## Dataset
- The dataset consists of CAPTCHA images stored in a specified directory.
- Each image filename (excluding extension) represents the CAPTCHA text label.
- Images are resized to `100x50` pixels and converted to grayscale before training.

## Model Architecture
- **Convolutional Layers**: Extract spatial features from CAPTCHA images.
- **MaxPooling Layers**: Reduce feature map size and improve generalization.
- **Flatten Layer**: Converts 2D features into a 1D vector.
- **Fully Connected Layers**: Map features to character predictions.
- **Softmax Activation**: Provides per-character classification probabilities.
- **Reshape Layer**: Ensures correct output format for multi-character prediction.

## Installation & Dependencies
Make sure you have the following dependencies installed:
```bash
pip install numpy opencv-python tensorflow scikit-learn
```

## Usage
### 1. Train the Model
Modify `data_dir` to point to your dataset folder and run:
```bash
python train.py
```
The trained model will be saved as `captcha_model_improved.keras`.

### 2. Predict CAPTCHA from an Image
```python
from predict import predict_captcha

captcha_image = "path_to_captcha.png"
result = predict_captcha(captcha_image)
print("Predicted CAPTCHA:", result)
```

## File Structure
```
📂 CAPTCHA-Recognition
├── train.py  # Training script for CNN model
├── predict.py  # Script to load model and predict CAPTCHA
├── dataset/  # Folder containing CAPTCHA images
├── captcha_model_improved.keras  # Saved model after training
├── README.md  # Project documentation
```

## Future Enhancements
- Improve accuracy with more complex CNN architectures (e.g., LSTMs, Attention Mechanisms).
- Implement data augmentation to handle noise and distortion.
- Extend support for variable-length CAPTCHAs.

## License
This project is open-source under the MIT License.

