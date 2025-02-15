# CaptchaCrackerAI
DeepCaptchaSolver is a CNN-based CAPTCHA solver that recognizes alphanumeric characters (A-Z, a-z, 0-9) from 100x50 grayscale images. Built with TensorFlow, OpenCV, and NumPy, it supports training, saving, and real-time predictions using one-hot encoding.

This project demonstrates how to build a Convolutional Neural Network (CNN) model to recognize and decode CAPTCHA images. CAPTCHAs are commonly used to distinguish between humans and bots, and this project aims to automate the process of solving them using deep learning techniques.

## Overview

The project involves the following steps:
1. **Data Loading and Preprocessing**: Load CAPTCHA images from a dataset, preprocess them (resize, normalize, and convert to grayscale), and encode the labels for training.
2. **Model Building**: Construct a CNN model using TensorFlow and Keras to predict the characters in the CAPTCHA images.
3. **Training**: Train the model on the preprocessed dataset and save the trained model for future use.
4. **Prediction**: Use the trained model to predict the text in new CAPTCHA images.

## Features
- **Character Set**: Supports recognition of alphanumeric characters (A-Z, a-z, 0-9).
- **Customizable CAPTCHA Length**: The model is designed to handle CAPTCHAs of a fixed length (default is 5 characters).
- **Improved CNN Architecture**: The model uses convolutional layers, max pooling, dropout for regularization, and a custom output layer to predict multiple characters simultaneously.
- **Easy-to-Use Prediction Function**: A simple function is provided to predict the text in a CAPTCHA image.

## Dataset
The dataset consists of CAPTCHA images and their corresponding labels (text). Each image is preprocessed and resized to a fixed dimension (100x50 pixels) before being fed into the model.

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `opencv-python`
- `tensorflow`
- `scikit-learn`

You can install the dependencies using:
```bash
pip install numpy opencv-python tensorflow scikit-learn
