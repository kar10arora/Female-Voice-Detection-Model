# Female-Voice-Detection-Model
# 1. Introduction

The goal of this project is to develop an emotion detection system that recognizes emotions from female voice recordings. The system uses deep learning techniques, specifically a Convolutional Neural Network (CNN)-based model, trained on extracted audio features.

# 2. Dataset and Preprocessing

## 2.1 Dataset

The dataset used is the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

It contains speech samples labeled with emotions such as neutral, calm, happy, sad, angry, fearful, disgusted, and surprised.

## 2.2 Preprocessing

Loading Audio Files:

Audio files are read using Librosa with a sample rate of 22050 Hz.

Convert stereo to mono if necessary.

Feature Extraction:

Mel-frequency cepstral coefficients (MFCCs) - Capture spectral properties of audio.

Chroma Features - Represent harmonic and tonal characteristics.

Mel Spectrogram - Represents energy distribution across different frequencies.

Spectral Contrast - Identifies amplitude variations in frequency bins.

Label Encoding:

The categorical emotion labels are converted to numerical form using LabelEncoder.

Data Augmentation (Optional):

Noise addition, pitch shifting, and time stretching are used to increase dataset variability.

# 3. Model Architecture

The model is a CNN-based deep learning architecture, which is structured as follows:

Input Layer

Accepts extracted feature vectors.

Convolutional Layers

Three Conv2D layers with ReLU activation and max-pooling to extract deep features.

Flatten Layer

Converts feature maps into a single feature vector.

Dense Layers

Fully connected layers for classification.

Output Layer

Softmax activation function to predict one of the 8 emotion categories.

# 4. Training the Model

The model is compiled with Adam optimizer.

Categorical cross-entropy is used as the loss function.

Performance metrics include accuracy and confusion matrix analysis.

# 5. Implementation - Real-time Emotion Detection

The trained model is used in a GUI application that can process uploaded or recorded voice clips.

## 5.1 Checking for Female Voice

The pitch of the voice is estimated using Librosa's pYIN algorithm.

If the average pitch is below 165 Hz, the voice is considered male, and an error is shown.

## 5.2 Emotion Prediction Process

Upload or record an audio file.

Feature extraction from the audio clip.

Pass the extracted features to the trained model.

Display the predicted emotion in the GUI.

# 6. Graphical User Interface (GUI) Implementation

Implemented using Tkinter.

Features:

Upload a WAV file or Record a voice clip.

Display the predicted emotion.

Handle male voice detection and prevent incorrect predictions.

# 7. Model Deployment

The model and its weights are saved as JSON and H5 files.

The trained model is loaded in the GUI script using TensorFlow's model_from_json.

# 8. Future Enhancements

Extend the model to handle male voices.

Implement real-time streaming emotion detection.

Improve dataset size with additional diverse voices.

Deploy as a web application using Flask or FastAPI.

# 9. Conclusion

This project successfully builds an emotion recognition system for female voices using deep learning. It extracts meaningful features from speech and classifies emotions using a CNN-based model, integrated into a user-friendly GUI application for real-time use.
