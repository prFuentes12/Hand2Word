# 🧠🤟 Hand2Word

**Hand2Word** is a real-time sign language recognition system that leverages computer vision and deep learning to interpret **American Sign Language (ASL)** gestures and convert them into text or speech.

The main goal of this project is to **bridge the communication gap** between hearing and hearing-impaired individuals by making sign language recognition more accessible and practical through technology.

---

## 🎯 Objective

To develop a high-accuracy Convolutional Neural Network (CNN) model that recognizes ASL hand gestures (excluding letters J and Z, which involve motion), and lay the foundation for a real-time system capable of translating signs into readable or audible formats.

---

## 🧩 Applications

- **Assistive Communication:** Helping individuals with hearing or speech impairments communicate effectively.
- **Education:** Interactive tool for learning and practicing sign language.
- **Customer Support:** Integration into kiosks or virtual assistants for inclusive communication.
- **Public Services:** Use in hospitals, government offices, and customer service desks.

---

## 🛠️ How It Works

The project follows these major steps:

1. **Data Setup:**
   - Downloads and prepares the Sign Language MNIST dataset from KaggleHub.
   - Organizes the data and performs basic Exploratory Data Analysis (EDA).

2. **Data Preprocessing:**
   - Normalizes and reshapes the image data (28x28 grayscale).
   - Applies data augmentation techniques (rotation, zoom, shifts) to improve model generalization.

3. **Model Architecture:**
   - A custom CNN with multiple convolutional, pooling, dropout, and dense layers.
   - Uses L2 regularization and dropout to reduce overfitting.

4. **Training:**
   - Trains the model for 15 epochs using the Adam optimizer and categorical crossentropy.
   - Achieves:
     - **Training Accuracy:** 95.43%
     - **Test Accuracy:** 93.45%

5. **Evaluation and Visualization:**
   - Plots training/validation accuracy and loss curves.
   - Saves the final model for later deployment.

---

## 📈 Results

| Metric            | Accuracy |
|-------------------|----------|
| **Training Set**  | 95.43%   |
| **Test Set**      | 93.45%   |

The model shows strong performance and generalizes well to unseen test data.

---

## 🚀 Future Work

- 🔤 **Real-time Prediction:** Integrate with a webcam for live gesture recognition.
- 🗣️ **Text-to-Speech Integration:** Convert recognized signs into spoken words.
- 🌐 **Multilingual Support:** Extend beyond ASL to support other sign languages.
- 🧠 **Transfer Learning:** Apply pretrained models for better accuracy and efficiency.
- 📱 **Mobile/Web App:** Deploy the system for real-world use via an app or website.

---

## 💡 Inspiration

Sign language is a powerful form of communication. With the help of AI, we can make it more inclusive, accessible, and integrated into modern digital life.
