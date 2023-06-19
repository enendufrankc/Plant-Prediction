Plant Disease Classification using TensorFlow
This project aims to classify plant diseases using deep learning with TensorFlow. It utilizes a convolutional neural network (CNN) model to classify images of plants into different disease categories.

Dataset
The dataset used for this project is the "PlantVillage" dataset. It consists of images of plants belonging to three different classes: Potato Early Blight, Potato Late Blight, and Tomato Bacterial Spot.

Setup and Dependencies
To run this project, you need to have TensorFlow and Matplotlib installed. You can install them using the following command:

python
Copy code
pip install tensorflow matplotlib
Usage
Import the required libraries:
python
Copy code
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
Set the necessary parameters:
python
Copy code
image_size = 256
batch_size = 32
channel = 3
epoch = 5
Load the dataset:
python
Copy code
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(image_size, image_size),
    batch_size=batch_size
)
Split the dataset into training, validation, and testing sets:
python
Copy code
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    # Function definition here

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
Preprocess and augment the data:
python
Copy code
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(image_size, image_size),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
Define the CNN model architecture:
python
Copy code
model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(batch_size, image_size, image_size, channel)),
    # Model architecture definition here
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=(batch_size, image_size, image_size, channel))
model.summary()
Compile and train the model:
python
Copy code
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=epoch,
    batch_size=batch_size,
    verbose=1,
    validation_data=val_ds
)
Evaluate the model:
python
Copy code
scores = model.evaluate(test_ds)
Plot training and validation metrics:
python
Copy code
# Plotting code here
Make predictions on new images:
python
Copy code
def predict(model, img):
    # Prediction code here

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        # Prediction code here
Save the model:
python
Copy code
model_version = max([int(i) for i in os.listdir("../saved_models") + [0]]) + 1
model.save(f"../saved_models/{model_version}")
Summary
This project demonstrates how to build a CNN model using TensorFlow for plant disease classification. It covers steps such as data preprocessing
