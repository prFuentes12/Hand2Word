import cv2
import numpy as np
from tensorflow.keras.models import load_model
import string

# 1. Cargar modelo
model = load_model("sign_language_model.h5")  # cambia a tu ruta

# 2. Mapeo de índices a letras
labels = list(string.ascii_uppercase)
labels.remove('J')
labels.remove('Z')

# 3. Función para preprocesar imagen igual que entrenamiento
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# 4. Iniciar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar una caja guía (donde poner la mano)
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Recortar y predecir
    roi = frame[y1:y2, x1:x2]
    input_data = preprocess_image(roi)
    prediction = model.predict(input_data)
    letter_index = np.argmax(prediction)
    predicted_letter = labels[letter_index]

    # Mostrar resultado
    cv2.putText(frame, f'Pred: {predicted_letter}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
