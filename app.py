import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load pre-trained ResNet50 model + higher level layers for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Define the feature extraction model
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Extract and normalize features
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Prepare filenames list from the images folder
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Extract and store features for each image
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save the extracted features and filenames using pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)
