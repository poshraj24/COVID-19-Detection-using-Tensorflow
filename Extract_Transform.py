import os
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Set dataset path
dataset_path = '/content/drive/My Drive/DataSet/'

# Define paths for the datasets
COVID_FOLDER = os.path.join(dataset_path, 'COVID')
NON_COVID_FOLDER = os.path.join(dataset_path, 'non-COVID')

# Function to load and preprocess images
def load_image_tf(image_path, img_size=(224, 224)):
    image = tf.io.read_file(image_path)  # Read image from file
    image = tf.image.decode_png(image, channels=3)  # Decode PNG (or use decode_jpeg if necessary)
    image = tf.image.resize(image, img_size)  # Resize the image to the target size
    image = image / 255.0  # Normalize to [0, 1] range
    return image

# Custom Dataset Generator
class CovidDataset(tf.keras.utils.Sequence):
    def __init__(self, COVID_FOLDER, NON_COVID_FOLDER, batch_size=32, img_size=(224, 224), augment=False):
        self.COVID_FOLDER = COVID_FOLDER
        self.NON_COVID_FOLDER = NON_COVID_FOLDER
        self.covid_images = glob.glob(COVID_FOLDER + '/*.png')
        self.non_covid_images = glob.glob(NON_COVID_FOLDER + '/*.png')
        self.labels = [1]*len(self.covid_images) + [0]*len(self.non_covid_images)
        self.images = self.covid_images + self.non_covid_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

        # Shuffle data
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images[:], self.labels[:] = zip(*combined)

        # Augmentation if required
        self.datagen = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=20
        ) if augment else None

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images, labels = [], []
        for i, path in enumerate(batch_x):
            img = load_image_tf(path, self.img_size)  # Updated to use load_image_tf
            if self.augment:
                img = self.datagen.random_transform(img.numpy())  # Apply augmentation if enabled
            images.append(img)
            labels.append(batch_y[i])
            
        return np.array(images), np.array(labels)

# Initialize the dataset
batch_size = 32
dataset = CovidDataset(COVID_FOLDER, NON_COVID_FOLDER, batch_size=batch_size, augment=True)

# Split dataset into training and testing (80% train, 20% test)
train_size = int(0.8 * len(dataset.images))
train_images = dataset.images[:train_size]
train_labels = dataset.labels[:train_size]
test_images = dataset.images[train_size:]
test_labels = dataset.labels[train_size:]

# Create tf.data.Dataset for training and testing
def create_tf_dataset(images, labels, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    # Map the loading and preprocessing function to the dataset
    dataset = dataset.map(lambda x, y: (load_image_tf(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size)

train_dataset = create_tf_dataset(train_images, train_labels, batch_size, shuffle=True)
test_dataset = create_tf_dataset(test_images, test_labels, batch_size, shuffle=False)
