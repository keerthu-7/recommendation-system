import kagglehub
import os

# Download the dataset using kagglehub to a specific folder
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small", path=r"D:\Fashion-Recommendation-System\fashion-dataset")

# Print the path where the dataset was downloaded
print("Path to dataset files:", path)

# Define the images directory within the downloaded path
image_dir = os.path.join(path, "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Move all downloaded images to the defined 'images' directory
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            # Ensure source and destination are not the same to avoid unnecessary moves
            src_path = os.path.join(root, file)
            dest_path = os.path.join(image_dir, file)
            if src_path != dest_path:
                os.rename(src_path, dest_path)

print(f"All images moved to: {image_dir}")
