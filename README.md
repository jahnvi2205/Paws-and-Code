# Paws-and-Code  
*A CNN-Based Cat and Dog Classifier*

## Overview  
WhiskerVision is a deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN). Built with Python and TensorFlow/Keras, this project demonstrates the power of CNNs in image classification tasks.

## Features  
- Accurately classifies images as either **Cat** or **Dog**.  
- Trained on a subset of the popular Cat-Dog dataset.
- Includes image resizing, normalization, and efficient data handling for seamless training. 
- Accepts custom images for testing the model's ability to classify unseen data.
- Implements techniques like dropout layers to reduce overfitting.

## Tech Stack  

The project utilizes the following technologies and libraries:  

1. **TensorFlow and Keras**  
   - **Framework**: TensorFlow  
   - **Model Building**: Keras API for creating and training the CNN model.  
   - **Key Modules**:  
     - `tensorflow` and `keras`: For defining and managing the neural network.  
     - **Layers Used**:  
       - `Dense`: Fully connected layer for classification.  
       - `Conv2D`: Convolutional layers to extract image features.  
       - `MaxPooling2D`: Down-sampling to reduce spatial dimensions.  
       - `Flatten`: Converts 2D matrix to 1D array for Dense layers.  
       - `BatchNormalization`: Stabilizes and accelerates training.  
       - `Dropout`: Prevents overfitting.  

2. **Matplotlib**  
   - **Visualization**: Used for plotting training and validation metrics (accuracy, loss).  

3. **OpenCV (cv2)**  
   - **Image Processing**:  
     - Reads and preprocesses images.  
     - Resizes images to ensure consistency in input size for the CNN model.  
