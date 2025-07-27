import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load product names from CSV
styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')
styles_df['Filename'] = styles_df['id'].astype(str) + ".jpg"
filename_to_name = pd.Series(styles_df.productDisplayName.values, index=styles_df.Filename).to_dict()

# Load the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def show_recommendations(indices, filenames, filename_to_name):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Debugging: Print filenames for the recommended indices
    for i in range(5):
        st.write(f"Recommended filename {i+1}: {filenames[indices[0][i]]}")

    with col1:
        st.image(filenames[indices[0][0]])
        st.caption(filename_to_name.get(os.path.basename(filenames[indices[0][0]]), "Unknown"))
    with col2:
        st.image(filenames[indices[0][1]])
        st.caption(filename_to_name.get(os.path.basename(filenames[indices[0][1]]), "Unknown"))
    with col3:
        st.image(filenames[indices[0][2]])
        st.caption(filename_to_name.get(os.path.basename(filenames[indices[0][2]]), "Unknown"))
    with col4:
        st.image(filenames[indices[0][3]])
        st.caption(filename_to_name.get(os.path.basename(filenames[indices[0][3]]), "Unknown"))
    with col5:
        st.image(filenames[indices[0][4]])
        st.caption(filename_to_name.get(os.path.basename(filenames[indices[0][4]]), "Unknown"))

# File upload handling
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        try:
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, feature_list)
            show_recommendations(indices, filenames, filename_to_name)
        except Exception as e:
            st.error(f"Error during feature extraction or recommendation: {e}")
    else:
        st.header("Some error occurred in file upload")
