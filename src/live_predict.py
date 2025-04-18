import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize MediaPipe and the TensorFlow model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the sign language model (make sure you have the trained model)
model = tf.keras.models.load_model("sign_language_model.h5") 

# List of labels excluding 'J' and 'Z'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # List excluding 'J' and 'Z'

# Initialize the camera
cap = cv2.VideoCapture(0)

prev_hand_detected = False
last_predicted_letter = None  # Initialize the last detected letter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally for better visualization
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB for processing with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Detect hands
    hand_detected = False
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Define the Region of Interest (ROI) for the prediction (the hand)
            x_min = min([landmark.x for landmark in landmarks.landmark])
            x_max = max([landmark.x for landmark in landmarks.landmark])
            y_min = min([landmark.y for landmark in landmarks.landmark])
            y_max = max([landmark.y for landmark in landmarks.landmark])
            
            h, w, _ = frame.shape
            roi = frame[int(y_min * h):int(y_max * h), int(x_min * w):int(x_max * w)]
            
            if roi.size > 0:
                hand_detected = True

                # Preprocess the image (convert to grayscale)
                roi_resized = cv2.resize(roi, (28, 28))  # Resize to 28x28
                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                
                # Normalize the image to values between 0 and 1
                gray = gray.astype("float32") / 255.0

                # Expand dimensions so that the model can process it (add batch dimension)
                gray = np.expand_dims(gray, axis=-1)  # Add an extra channel dimension
                gray = np.expand_dims(gray, axis=0)   # Add an extra dimension for batch

                # Perform prediction with the model
                predictions = model.predict(gray)
                letter_index = np.argmax(predictions)

                # Check if the index is valid before accessing 'labels'
                if 0 <= letter_index < len(labels):
                    predicted_letter = labels[letter_index]

                    # Print only if there is a change in the detected letter
                    if predicted_letter != last_predicted_letter:
                        print(f"Detected letter: {predicted_letter}")
                        last_predicted_letter = predicted_letter  # Update the last detected letter

                # Display the prediction on the screen
                cv2.putText(frame, f"Prediction: {predicted_letter}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Print hand detection status only if it changes
    if hand_detected != prev_hand_detected:
        if hand_detected:
            print("✅ Hand detected.")
        else:
            print("❌ No hand detected.")
        prev_hand_detected = hand_detected

    # Display the image
    cv2.imshow("Hand Tracking", frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
