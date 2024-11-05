import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carregar o modelo pré-treinado
model_best = load_model("C:\\Users\\anton\\OneDrive\\Documentos\\Data Projects\\ExpressaoFacial\\face_model.h5") # set your machine model file path here

# Classes para mapear as expressões faciais
class_names = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutro']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera (0 is usually the default camera)
cap = cv2.VideoCapture(3)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Converter para escala de cinza para face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Processa cada face detectada
    for (x, y, w, h) in faces:
        # Extrai a reagião da face
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        # Fazer a predição da expressão facial usando o modelo
        predictions = model_best.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]

        # Descrever a emoção detectada
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

        # Desenhar um retângulo ao redor do rosto 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mostrar o frame com a detecção
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()