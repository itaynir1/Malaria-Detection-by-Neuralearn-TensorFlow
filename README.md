# Malaria Detection by Neuralearn.ai
This project aims to detect malaria using a deep learning model implemented in TensorFlow. The model is based on a modified LeNet architecture and is trained on a dataset of malaria images.

## Getting Started
To run this project, you need the following dependencies:

- TensorFlow
- NumPy
- Matplotlib
- TensorFlow Datasets
- Google Colab (for the Colaboratory environment)

## Project Structure
The main file for this project is "Malaria Detection by Neuralearn.ai.ipynb".

It contains the code for downloading the dataset, visualizing the data, preprocessing it, creating the model, training, evaluation, and testing.

## Dataset
The malaria dataset is fetched using TensorFlow Datasets. The dataset is split into training, validation, and testing sets with ratios of 80%, 10%, and 10%, respectively.

## Data Visualization
Before preprocessing, some images from the training dataset are visualized using Matplotlib to get a better understanding of the data.

## Data Preprocessing
The images are resized and rescaled to a fixed size of 224x224 pixels and normalized to values between 0 and 1.

## Model Creation
The model architecture is based on the LeNet architecture with some modifications. It consists of convolutional layers, batch normalization layers, max-pooling layers, and fully connected layers with a sigmoid activation function in the output layer for binary classification.

## Model Training
The model is trained using the Adam optimizer and Binary Crossentropy loss function for 20 epochs. The training and validation accuracy and loss are plotted for visualization.

## Model Evaluation and Testing
The trained model is evaluated on the testing dataset to measure its performance. The evaluation metrics include loss and accuracy. Additionally, some sample images from the test dataset are visualized along with their predicted classes.

## Loading and Saving the Model
The model is saved in two formats: SavedModel and HDF5 format. Additionally, the model's weights are saved separately in a `"weights"` folder.

## Google Drive Integration
The project uses Google Drive to mount the Colaboratory environment for easy access to files and models.

## How to Use the Code
1. Make sure you have all the required dependencies installed.
2. Open the "Malaria Detection by Neuralearn.ai.ipynb" notebook in a Jupyter environment or Google Colab.
3. Run the notebook cells sequentially to download the dataset, preprocess it, create and train the model, evaluate it, and visualize the results.

**Note:** If you encounter any issues with dependencies or dataset download, please refer to the original Colab file link provided at the beginning of the notebook.
