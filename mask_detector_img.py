import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Carregar o modelo treinado
model = load_model('keras_model.h5')

# Função para pré-processar a imagem do rosto
def preprocess_face(face, target_size=(224, 224)):
    face_resized = cv2.resize(face, target_size)  # Redimensiona para o tamanho correto
    face_resized = np.expand_dims(face_resized, axis=0)  # Expande para incluir a dimensão de lote
    face_resized = face_resized / 255.0  # Normaliza para [0, 1]
    return face_resized

# Função para processar a imagem estática e mostrar na interface Tkinter
def process_image(image_path, image_label):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Converter a imagem para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicializar MediaPipe para detecção de rosto
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Inicializando MediaPipe face detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Processar a imagem para detectar rostos
        results = face_detection.process(image_rgb)

        # Se rostos forem detectados
        if results.detections:
            for detection in results.detections:
                # Desenhar as caixas ao redor dos rostos detectados
                mp_drawing.draw_detection(image_rgb, detection)  # Usar image_rgb aqui

                # Obter a localização da caixa delimitadora
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Extrair o rosto da imagem
                face = image[y:y+h, x:x+w]

                if face.size > 0:
                    try:
                        # Pré-processar o rosto detectado
                        face_preprocessed = preprocess_face(face)

                        # Fazer a previsão de máscara/no máscara
                        prediction = model.predict(face_preprocessed)
                        print("Prediction raw output:", prediction)

                        # Verificar a previsão e definir a label
                        mask_prob = prediction[0][0]  # Supondo que a saída do modelo é [mask_prob, no_mask_prob]
                        label = "Mask On" if mask_prob > 0.5 else "Mask Off"

                        # Desenhar a label no frame
                        cv2.putText(image_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Erro ao processar o rosto: {e}")

    # Converter a imagem para exibir na interface Tkinter
    img_pil = Image.fromarray(image_rgb)  # Use image_rgb aqui
    img_pil.thumbnail((400, 400), Image.LANCZOS)

    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Exibir a imagem dentro da interface
    image_label.imgtk = img_tk
    image_label.config(image=img_tk)
