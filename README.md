
# UrbanSound8K Audio Classification
# Introduction
This project demonstrates the application of Convolutional Neural Networks (CNN) for classifying urban sound events using audio data. The model is trained on the UrbanSound8K dataset, which contains audio samples of various urban sounds. The CNN model is designed to identify the type of sound based on the audio features extracted from the samples.

# Dataset
The dataset used in this project is the UrbanSound8K dataset. It contains audio samples categorized into different classes based on the type of sound. The dataset is divided into training, validation, and test sets.
https://www.kaggle.com/datasets/chrisfilo/urbansound8k


# Data Preprocessing
Loading and Visualizing Data
The audio samples are loaded using the librosa library. The audio data is converted to mono and visualized using matplotlib to understand the waveform characteristics.

# Extracting Features
Mel-Frequency Cepstral Coefficients (MFCC) are extracted from the audio samples. The MFCC summarizes the frequency distribution across the window size, allowing analysis of both frequency and time characteristics of the sound.

# Splitting the Dataset
The dataset is split into training, validation, and test sets. This ensures a balanced distribution of samples for model evaluation.

# Model Building
Model Architecture
The model architecture is built using a sequential CNN. It includes the following layers:

Dense Layers: Multiple fully connected layers with ReLU activations.
Dropout Layers: Prevents overfitting by randomly dropping units during training.
Final Dense Layer: Performs classification using a softmax activation.
Compiling the Model
The model is compiled using the Adam optimizer, categorical crossentropy loss function, and accuracy as the evaluation metric.

# Training the Model
The model is trained for 100 epochs using the training dataset. The validation dataset is used to monitor the model's performance and prevent overfitting. The best model is saved using the ModelCheckpoint callback.

# Model Evaluation
The trained model is evaluated on the test dataset to measure its accuracy. The accuracy and loss curves are plotted to visualize the training and validation performance over the epochs.

# Inference
A function is created to perform inference on new audio samples. The function extracts features from the audio, reshapes them for the model, and predicts the class of the sound. The predicted label is inverse-transformed to obtain the original class name.

# Conclusion
This README provides an overview of the process of building, training, and evaluating a CNN model for urban sound classification. The model achieves high accuracy and is capable of accurately identifying various urban sounds from the audio samples.# 