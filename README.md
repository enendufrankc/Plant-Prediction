
#Plant Disease Classification with Convolutional Neural Networks (CNN)

##Overview
This Python script is a machine learning application using TensorFlow and Keras libraries to create a Convolutional Neural Networks (CNN) model. This model is designed to classify images of plants into three categories based on the presence of disease and type of disease.

The script works by training the CNN model on a dataset of plant images found in the 'PlantVillage' directory, partitioning the data into training, validation, and testing sets. The images are processed and augmented before they are fed into the model. The model is then trained for a specified number of epochs.

Finally, the script evaluates the trained model on the testing set, and demonstrates how to use the trained model to make a prediction on a single image. The trained model is then saved in the 'saved_models' directory.

##Usage
Ensure that you have the necessary packages installed, these include TensorFlow and matplotlib among others.

Prepare your image dataset. This script expects the images to be stored in a directory named "PlantVillage", and the subdirectories in this folder should be the class names.

Run the script in a Python environment where you have TensorFlow and other necessary packages installed. You can run this script in an environment like Jupyter Notebook or Google Colab.

The model's training and validation accuracy/loss during the training process is displayed in real-time, and a graph showing these metrics across all epochs will be shown once training is complete.

The script will also output the model's accuracy on the test dataset.

An example image from the test set will be used for prediction to demonstrate how the model performs on unseen data.

Finally, the trained model will be saved both in TensorFlow SavedModel format in the 'saved_models' directory and in HDF5 format as 'potatoes.h5' in the project's root directory.

##Code Structure
The script begins by setting up the necessary parameters and loading the image dataset from the 'PlantVillage' directory.
It then visualizes a batch of images from the dataset.
Next, it partitions the dataset into training, validation, and testing sets.
The model is then built using a sequential Keras model, which includes several convolutional and max-pooling layers, followed by dense layers.
The model is compiled and then trained on the training set for a specified number of epochs.
After training, the model's performance is evaluated on the testing set.
The script demonstrates how to use the trained model to make predictions on individual images.
Finally, the trained model is saved in the 'saved_models' directory and as 'potatoes.h5'.

##Contributing
This script is open-source, and contributions are welcome. If you would like to contribute, you can open a pull request with your proposed changes. Please ensure that any additions or modifications are well-documented.
