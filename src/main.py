import kagglehub
from PIL import Image
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt


# ==========================================
# CREATE FOLDER (IF NOT EXISTS)
# ==========================================

# Get the parent directory path (outside 'src')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Define the 'data' folder path in the parent directory
data_dir = os.path.join(parent_dir, 'data')

# Create the 'data' folder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Folder created: {data_dir}")
else:
    print(f"The folder already exists: {data_dir}")


# ==========================================
# DOWNLOAD DATA 
# ==========================================

# Download dataset from KaggleHub to the default directory
dataset_path = kagglehub.dataset_download("datamunge/sign-language-mnist")

# Move the downloaded files to the 'data' folder
# Assuming the downloaded file is compressed (e.g., .zip)
downloaded_files = os.listdir(dataset_path)

# Loop through the downloaded files and move them to the 'data' folder
for file in downloaded_files:
    source_path = os.path.join(dataset_path, file)
    destination_path = os.path.join(data_dir, file)
    shutil.move(source_path, destination_path)

print(f"Files moved to {data_dir}")

# Load CSVs from the 'data' folder
train_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))