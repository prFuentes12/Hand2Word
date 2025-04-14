import kagglehub
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns


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


# ==========================================
# EDA
# ==========================================

# Print first 5 rows of the train dataset
print("First 5 rows of the train dataset:")
print(train_df.head())

# Print first 5 rows of the test dataset
print("\nFirst 5 rows of the test dataset:")
print(test_df.head())

# Define the label mapping, excluding J and Z
label_map = {i: chr(ord('A') + i) for i in range(26) if i not in [9, 25]}

# Show the distinct values in the 'label' column along with their corresponding letters
distinct_labels = train_df['label'].unique()

# Print the label and corresponding letter
for label in distinct_labels:
    print(f"Label: {label} -> Letter: {label_map[label]}")


# Check for missing values
def check_missing_values(df, df_name):
    print(f"Missing values in {df_name}:")
    print(df.isnull().sum())  # Number of missing values in each column
    print(f"Percentage of missing values in {df_name}:")
    print(df.isnull().mean() * 100)  # Percentage of missing values
    print("\n")

check_missing_values(train_df, "train_df")
check_missing_values(test_df, "test_df")

# Show descriptive statistics of the data
def describe_data(df, df_name):
    print(f"Descriptive statistics for {df_name}:")
    print(df.describe())  # Descriptive statistics of the numeric columns
    print("\n")

describe_data(train_df, "train_df")
describe_data(test_df, "test_df")

# Class distribution in the labels column
def plot_class_distribution(df, df_name):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='Set2')  # Countplot to show class distribution
    plt.title(f"Class Distribution in {df_name}")
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.show()

plot_class_distribution(train_df, "train_df")
plot_class_distribution(test_df, "test_df")

# Display a few sample images from the dataset
def plot_sample_images(df, num_samples=6):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        label = df.iloc[i, 0]  # Get the label (letter)
        pixels = df.iloc[i, 1:].values.reshape(28, 28).astype("uint8")  # Reshape the pixel values
        plt.subplot(2, num_samples//2, i + 1)
        plt.imshow(pixels, cmap="gray")  # Display the image in grayscale
        plt.title(f"Label: {label}")
        plt.axis("off")  # Remove axis
    plt.tight_layout()
    plt.show()

plot_sample_images(train_df)

# Plot the distribution of pixel values in the dataset
def plot_pixel_value_distribution(df, df_name):
    pixel_columns = df.columns[1:]  # Ignore the label column, consider only pixel columns
    pixel_values = df[pixel_columns].values.flatten()  # Flatten all pixel values to a 1D array

    plt.figure(figsize=(8, 6))
    sns.histplot(pixel_values, bins=50, kde=True)  # Plot histogram with KDE (Kernel Density Estimate)
    plt.title(f"Pixel Value Distribution in {df_name}")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

plot_pixel_value_distribution(train_df, "train_df")

# Identify outliers in the pixel columns using boxplots
def plot_outliers(df, df_name):
    pixel_columns = df.columns[1:]  # Ignore the label column, consider only pixel columns
    plt.figure(figsize=(12, 8))
    
    for i, column in enumerate(pixel_columns[:6]):  # Display only a subset of columns for clarity
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x=df[column], color='lightblue')  # Boxplot to visualize outliers
        plt.title(f"Outliers in {df_name} - Pixel {column}")
        plt.xlabel('Pixel Value')
    
    plt.tight_layout()
    plt.show()

plot_outliers(train_df, "train_df")
