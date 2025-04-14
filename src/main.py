import os
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CREATE FOLDER (IF NOT EXIST)
# ==========================================

# Get the parent directory path (outside 'src')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Define the 'data' folder path in the parent directory
data_dir = os.path.join(parent_dir, 'data')

# Create the 'data' folder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)  # Create folder if it doesn't exist
    print(f"Folder created: {data_dir}")
else:
    print(f"The folder already exists: {data_dir}")


# ==========================================
# DOWNLOAD DATA 
# ==========================================

# Download dataset from KaggleHub to the default directory
dataset_path = kagglehub.dataset_download("datamunge/sign-language-mnist")

# Move the downloaded files to the 'data' folder
downloaded_files = os.listdir(dataset_path)

# Loop through the downloaded files and move them to the 'data' folder
for file in downloaded_files:
    source_path = os.path.join(dataset_path, file)  # Source path of the file
    destination_path = os.path.join(data_dir, file)  # Destination path in the 'data' folder
    shutil.move(source_path, destination_path)  # Move file to 'data' folder

print(f"Files moved to {data_dir}")

# Load CSVs from the 'data' folder
train_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))  # Load training data
test_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))    # Load test data


# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
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

# ==========================================
# DATA PREPARATION
# ==========================================

# Separate features and labels for train and test data
X_train = train_df.iloc[:, 1:].values   # All columns except the label
y_train = train_df.iloc[:, 0].values    # The label column

X_test = test_df.iloc[:, 1:].values     # All columns except the label
y_test = test_df.iloc[:, 0].values     # The label column

# Normalize the data by dividing by 255 (since pixel values range from 0 to 255)
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# Reshape data to 28x28x1 (grayscale image format)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Print the reshaped dimensions
print("X_train reshaped:", X_train.shape)
print("X_test reshaped:", X_test.shape)

# Determine the number of unique classes (labels)
num_classes = len(np.unique(y_train))+1
print("Number of classes:", num_classes)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Data Augmentation: Rotating, zooming, and shifting images
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)  # Fit the augmentation parameters to the training data

# -------------------------------
# DEFINING THE CNN MODEL
# -------------------------------

# Build a Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                  kernel_regularizer=regularizers.l2(0.001)),  # Conv layer with L2 regularization
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Dropout(0.25),  # Dropout layer to prevent overfitting

    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),  # Conv layer with L2 regularization
    layers.MaxPooling2D((2, 2)),  # Max pooling layer
    layers.Dropout(0.25),  # Dropout layer

    layers.Flatten(),  # Flatten the data for the fully connected layers
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),  # Dense layer
    layers.Dropout(0.5),  # Dropout layer
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # Print model summary


# -------------------------------
# TRAINING THE MODEL
# -------------------------------

# Train the CNN with the training data and validate with the test set
history = model.fit(
    X_train, 
    y_train_cat, 
    epochs=15,        #Number of epochs
    batch_size=64,    # Batch size
    validation_data=(X_test, y_test_cat)
)

model.save("sign_language_model.h5")  # Save the model to a file


# -------------------------------
# EVALUATING THE MODEL
# -------------------------------

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print("\nTest accuracy:", test_acc)


# -------------------------------
# PLOTTING RESULTS
# -------------------------------

# Plot accuracy and loss during training
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
