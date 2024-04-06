

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
import backend as _bk
import requests
import json

# ... (Previous code for data loading, feature extraction, and similarity search)

# Streamlit web app for data visualization and recommendations


def main():
    st.title("Fashion Recommendation App")
    st.sidebar.header("User Input")

    # User selects an image for fashion recommendation
    uploaded_image = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.sidebar.image(
            uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.sidebar.write("")
        st.sidebar.write("Generating Recommendations...")

        image = Image.open(uploaded_image)
        # st.image(image,width=300,caption="Image Uploaded")
        # Save the image to your local directory
        save_path = os.path.join(os.getcwd(), "saved_images")
        os.makedirs(save_path, exist_ok=True)

        image_filename = os.path.join(save_path, uploaded_image.name)
        image.save(image_filename)
        # Calculate recommendations based on the uploaded image
        recommendations = _bk.similar(image_filename, _bk.model1)
        print(recommendations)

        st.subheader("Recommended Fashion Items:")
        # Display recommended fashion items with their images
        for item_path in recommendations:
            item_image = Image.open(item_path)
            st.image(item_image, caption="Recommended Item")

        product_description_directory = r'product_description_directory'

      # Load the CSV  file into a DataFrame using pandas
        # Use regular expression to extract the number
        df = pd.read_csv(product_description_directory)
        match = re.search(r'(\d+).jpg$', recommendations[0])

        if match:
            number = match.group(1)
            product_description = df.loc[df['id'] == int(
                number), 'productDisplayName'].iloc[0]
            print(product_description)

            template = f"""
        you are a fashion recommendation stylist, 
         give only 4 style recommendations for this item, and do not mention the name of the item as well as its colour in your recommendation. Always Reply in json format
      "style": "recommendation":{product_description}

        """
            reqUrl = ''
            reqHeaders = {'Authorization': 'Bearer ' + ''}
            messagesToSend = [{"role": "user", "content": template}]
            reqBody = {
                "model": "gpt-3.5-turbo",
                "messages": messagesToSend,
                "temperature": 0,
            }

            response = requests.post(
                reqUrl, stream=True, headers=reqHeaders, json=reqBody)
            # Parse the JSON response
            parsed_response = json.loads(response.content)

            # Extract the response content
            response_content = parsed_response["choices"][0]["message"]["content"]

            # Decode the content to get the actual response
            actual_response = json.loads(response_content)

            # Print the extracted response
            print(actual_response["style"]["recommendation"])
            ss = actual_response["style"]["recommendation"]
            col1, col2 = st.columns(2)
            with col1:
                st.header("Option 1")
                st.write(ss[0])

            with col2:
                st.header("Option 2")
                st.write(ss[1])

            col3, col4 = st.columns(2)

            with col3:
                st.header("Option 3")
                st.write(ss[2])

            with col4:
                st.header("Option 4")
                st.write(ss[3])


if __name__ == "__main__":
    main()
