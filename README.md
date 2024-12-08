# CNN and Transfer Learning Image Classification

## Overview
This project focuses on image classification using Convolutional Neural Networks (CNN) and Transfer Learning with the VGG16 model. Three versions of the code are provided, each with enhancements and different approaches to improve model accuracy and performance.

## Dataset
https://www.kaggle.com/datasets/noshintasnia/brain-stroke-prediction-ct-scan-image-dataset
https://www.kaggle.com/datasets/alymaher/brain-stroke-ct-scan-image

The datasets are expected to be organized in the following structure:
Dataset/
    Train/
    Test/
    Validation/


## Project Structure
- **Code 1:** A simple CNN model.
- **Code 2:** Transfer Learning with the VGG16 model.
- **Code 3:** An enhanced CNN model with additional layers and regularization techniques.


Code 1: Simple CNN Model
- This code defines and trains a simple CNN model on the dataset.
- Layers: Conv2D, MaxPooling2D, Dense, Flatten
- Activation: ReLU for hidden layers, Sigmoid for the output layer
- Optimizer: Adam
- Loss Function: BinaryCrossentropy

Code 2: Transfer Learning with VGG16
- This code uses the VGG16 model pre-trained on ImageNet data.
- Base Model: VGG16
- Custom Layers: Dense, Dropout
- Optimizer: Adam with a lower learning rate
- Loss Function: BinaryCrossentropy

Code 3: Enhanced CNN Model
- This code enhances the simple CNN with additional layers, Batch Normalization, and Dropout for regularization.
- Layers: Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
- Activation: ReLU for hidden layers, Sigmoid for the output layer
- Optimizer: Adam
- Loss Function: BinaryCrossentropy

# Usage
Run the desired code (Code 1, Code 2, or Code 3) to train and evaluate the model. Each script includes:
Loading and normalizing the datasets.
Defining the model architecture.
Compiling the model.
Training the model with TensorBoard logging.
Evaluating the model on testing data.
Visualizing predictions and performance metrics.

# Visualizations
The scripts include code to visualize:
Predictions on a batch of testing data.
Training and validation accuracy and loss.

# Results
Each code version prints the test accuracy and loss after model evaluation. Visualizations of predictions and performance metrics are displayed.

## License

The Code 1 is licensed under the MIT License. 

link: https://opensource.org/licenses/MIT

