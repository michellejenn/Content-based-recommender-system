import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import tensorflow
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import pandas as pd
import re
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from PIL import Image


model1 = ResNet50(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))
model1.trainable = False
model1 = tensorflow.keras.Sequential([
    model1,
    GlobalMaxPooling2D()
])


def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = np.array(image)

        return image
    except:
        return None


# this function loads the images and also extracts the features from the image
def load_dataset(model):
    image_data = []
    file_name = []

    image_directory = r'/imagesfilepath'
    for filename in os.listdir(image_directory):
        if len(image_data) < 5000:
            if filename.endswith(".jpg") or filename.endswith(".png"):

                img = preprocess_image(os.path.join(image_directory, filename))
                expand_img = np.expand_dims(img, axis=0)
                pre_img = preprocess_input(expand_img)
                result = model.predict(pre_img).flatten()
                normalized = result/norm(result)
                image_data.append(normalized)
                file_name.append(os.path.join(image_directory, filename))

    pickle.dump(image_data, open("imageDat.pkl", 'wb'))
    pickle.dump(file_name, open("file_name.pkl", "wb"))
    return image_data

# load_dataset(model1)


def similar(image_path, model):
    image = preprocess_image(image_path)
    expand_img = np.expand_dims(image, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result/norm(result)
    imageData = pickle.load(open("imageDat.pkl", "rb"))
    file_name = pickle.load(open("file_name.pkl", "rb"))
    # product_description = pickle.load(open("product_description.pkl","rb"))
    neigh = NearestNeighbors(
        n_neighbors=5, algorithm="brute", metric="euclidean")
    neigh.fit(imageData)

    distance, indices = neigh.kneighbors([normalized])

   # Get the indices of the nearest neighbors
    nearest_neighbor_indices = indices[0]

    # Retrieve the values of the nearest neighbors
    nearest_neighbor_values = [imageData[i] for i in nearest_neighbor_indices]
    print(nearest_neighbor_values)

    # # If you have corresponding file names, retrieve them
    nearest_neighbor_file_names = [file_name[i]
                                   for i in nearest_neighbor_indices]

    print(nearest_neighbor_file_names)
    return nearest_neighbor_file_names
