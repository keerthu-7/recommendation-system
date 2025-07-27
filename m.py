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
import json

# --- Load user credentials from JSON ---
def load_user_credentials():
    try:
        with open("users.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# --- Save user credentials to JSON ---
def save_user_credentials(credentials):
    with open("users.json", "w") as file:
        json.dump(credentials, file)

# --- Initialize user credentials in session state ---
if "user_credentials" not in st.session_state:
    st.session_state["user_credentials"] = load_user_credentials()

# --- Add a new user ---
def add_user(username, password):
    if username in st.session_state["user_credentials"]:
        st.warning("⚠ Username already exists. Please choose another one.")
    else:
        st.session_state["user_credentials"][username] = password
        save_user_credentials(st.session_state["user_credentials"])
        st.success("🎉 Account created successfully! You can now log in.")

# --- Validate login credentials ---
def validate_login(username, password):
    return st.session_state["user_credentials"].get(username) == password

# --- LOGIN INTERFACE ---
def show_login_page():
    st.markdown("<div class='main-title'>👗 Fashion Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Welcome! Please log in or create an account to continue.</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Create New Account"])

    with tab1:
        st.subheader("🔑 Log In")
        login_username = st.text_input("Username", key="login_username", help="Enter your existing username.")
        login_password = st.text_input("Password", type="password", key="login_password", help="Enter your password.")
        login_button = st.button("Log In", key="login_button", use_container_width=True)

        if login_button:
            if validate_login(login_username, login_password):
                st.success("✅ Logged in successfully!")
                # Set session state to indicate the user is logged in
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_username
                st.session_state["page"] = "main"  # Set the page to main
            else:
                st.error("❌ Invalid username or password. Please try again.")

    with tab2:
        st.subheader("🆕 Create New Account")
        new_username = st.text_input("New Username", key="new_username", help="Choose a unique username.")
        new_password = st.text_input("New Password", type="password", key="new_password", help="Create a strong password.")
        create_button = st.button("Create Account", key="create_button", use_container_width=True)

        if create_button:
            add_user(new_username, new_password)

# --- MAIN APP CONTENT (from main.py) ---
def show_main_app():
    st.title("Welcome to the Fashion Recommendation System!")

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

    st.subheader("Fashion Recommendations")

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

        def generate_myntra_link(filename):
            product_id = os.path.splitext(os.path.basename(filename))[0]  # Extract the product ID from the filename
            return f"https://www.myntra.com/{product_id}"

        # Iterate over columns and display image, product name, and Myntra link
        for i, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                st.image(filenames[indices[0][i]])
                product_name = filename_to_name.get(os.path.basename(filenames[indices[0][i]]), "Unknown")
                st.caption(product_name)
                myntra_link = generate_myntra_link(filenames[indices[0][i]])
                st.markdown(f"[View on Myntra]({myntra_link})", unsafe_allow_html=True)


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

    # Example of a logout button
    if st.button("Logout"):
        # Reset the session state to log out the user
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["page"] = "login"

# --- APP START ---
# Initialize session state for login if it does not exist
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Set default page if not in session state
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# CSS Styling for a consistent theme
st.markdown("""
    <style>
        .main-title {
            color: #2E86C1; 
            font-size: 36px; 
            font-weight: bold;
        }
        .sub-title {
            color: #2874A6; 
            font-size: 20px;
        }
        .login-button, .create-button {
            background-color: #2E86C1; 
            color: white;
        }
        .login-button:hover, .create-button:hover {
            background-color: #2874A6;
        }
    </style>
""", unsafe_allow_html=True)

# Display either login or main app based on session state
if st.session_state["page"] == "login":
    show_login_page()
else:
    show_main_app()
