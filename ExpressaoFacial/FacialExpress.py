import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo pré-treinado (exemplo: modelo treinado em FER2013)
model = load_model("C:\\Users\\anton\\OneDrive\\Documentos\\Data Projects\\ExpressaoFacial\\face_model.h5")

# Dicionário para mapear as expressões faciais
expression_labels = {0: 'Raiva', 1: 'Nojo', 2: 'Medo', 3: 'Felicidade', 4: 'Tristeza', 5: 'Surpresa', 6: 'Neutro'}

# Inicializar a câmera
cap = cv2.VideoCapture(3)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Converter para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Carregar o classificador Haar para detecção de rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detectar rostos
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extrair o rosto detectado
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Fazer a predição da expressão facial
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        predicted_emotion = expression_labels[max_index]

        # Desenhar um retângulo ao redor do rosto e escrever a emoção detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Mostrar o frame com a detecção
    cv2.imshow('Expressão Facial', frame)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
