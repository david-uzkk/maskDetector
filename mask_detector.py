# mask_detector.py

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Carregar o modelo treinado
model = load_model('keras_model.h5')

# Função para pré-processar a imagem do rosto
def preprocess_face(face, target_size=(224, 224)):
    face_resized = cv2.resize(face, target_size)  # Redimensiona para 224x224
    face_resized = np.expand_dims(face_resized, axis=0)  # Expande a dimensão do lote
    face_resized = face_resized / 255.0  # Normaliza para o intervalo [0, 1]
    return face_resized

# Função para detectar a máscara via webcam e exibir no Tkinter
def mask_detector_webcam(video_label):
    # Inicializar MediaPipe para detecção de rosto
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Inicializar webcam
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Erro ao acessar a webcam.")
                break

            # Converter a imagem para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectar rostos
            results = face_detection.process(frame_rgb)

            # Processar cada rosto detectado
            if results.detections:
                for detection in results.detections:
                    # Obter a caixa delimitadora
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)

                    # Extrair o rosto da imagem
                    face = frame[y:y+h, x:x+w]

                    if face.size > 0:
                        # Pré-processar o rosto
                        face_preprocessed = preprocess_face(face)

                        # Fazer a previsão
                        prediction = model.predict(face_preprocessed)
                        mask_prob = prediction[0][0]  # Probabilidade de "com máscara"

                        # Definir a label
                        label = "Mask Off" if mask_prob > 0.5 else "Mask On"
                        color = (0, 0, 255) if mask_prob > 0.5 else (0, 255, 0)

                        # Desenhar a label e a caixa no rosto
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Converter a imagem para o formato que o Tkinter pode exibir
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)

            # Atualizar o label da imagem na interface
            video_label.imgtk = frame_tk
            video_label.config(image=frame_tk)

            # Permitir que o Tkinter atualize a interface
            video_label.update_idletasks()
            video_label.after(10, mask_detector_webcam, video_label)

        cap.release()
