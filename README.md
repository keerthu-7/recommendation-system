👗 Fashion Recommendation System

A deep learning–based fashion recommendation system that suggests visually similar clothing items from a dataset of 45,000+ fashion product images.
The system leverages transfer learning (ResNet-50) for feature extraction and an optimized K-Nearest Neighbors (Annoy) algorithm for efficient similarity search.

🚀 Features

Image-based recommendations → Upload an image to find the top 5 closest matches.

Deep Learning backbone → ResNet-50 pretrained on ImageNet for feature extraction.

Efficient similarity search → Annoy for fast nearest neighbor queries.

Interactive Web App → Built with Streamlit for a simple, user-friendly interface.

Automated Dataset Download → Script included to fetch dataset directly.

🛠️ Tech Stack

Languages: Python

Libraries/Frameworks: TensorFlow, Keras, scikit-learn, Annoy, NumPy, Pandas, Streamlit

Tools: Git, Jupyter Notebook, Kaggle

## 📂 Project Structure
fashion-recommendation/   <br>
│── app.py # Streamlit app (main UI for recommendations)  <br>
│── download_dataset.py # Script to download dataset  <br>
│── main.py # Main training / feature extraction pipeline  <br>
│── m.py # Model / utility functions  <br>
│── requirements.txt # Python dependencies   <br>
│── README.md # Project documentation   <br>



⚙️ Installation & Setup

1. Clone the repository



2. Create a virtual environment & install dependencies


3. Download the dataset
       Run the dataset download script:

        python download_dataset.py


4. Run the Streamlit app

      streamlit run m.py

🎯 Usage

Upload a fashion item image (e.g., shirt, dress, accessory).

The system extracts its features and finds the top 5 most similar products.

Results are displayed with images and similarity scores.

📊 Results

Extracted embeddings for 45,000+ images using ResNet-50.

Achieved fast retrieval (~0.01s/query) with Annoy similarity search.

Deployed interactive recommendations in a Streamlit web app.

🖼️ Demo

<img width="1038" height="651" alt="image" src="https://github.com/user-attachments/assets/f08d2fae-9305-49ad-87f0-1c8cf9034d14" />

<img width="1002" height="773" alt="image" src="https://github.com/user-attachments/assets/57b7b767-b4a8-4e6a-9957-e4e6e2542cc2" />



🔮 Future Improvements

Add collaborative filtering for hybrid recommendations.

Deploy on cloud (AWS/GCP/Heroku).

Extend dataset with more categories and higher-resolution images.

Enhance UI with filters (price, color, brand).




