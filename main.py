import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
import tensorflow as tf
import numpy as np
import mediapipe as mp
from mask_detector import mask_detector_webcam  # Importando o arquivo mask_detector.py

# Carregar o modelo treinado globalmente
model = tf.keras.models.load_model('keras_model.h5')

# Função para pré-processar a imagem do rosto
def preprocess_face(face, target_size=(224, 224)):
    try:
        face_resized = cv2.resize(face, target_size)  # Redimensiona para o tamanho correto
        face_resized = np.expand_dims(face_resized, axis=0)  # Expande para incluir a dimensão de lote
        face_resized = face_resized / 255.0  # Normaliza para [0, 1]
        return face_resized
    except Exception as e:
        print(f"Erro ao pré-processar o rosto: {e}")
        return None

# Função para processar a imagem e mostrar na interface Tkinter
def process_image(image_path, image_label):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Converter a imagem para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicializar MediaPipe para detecção de rosto
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Processar a imagem para detectar rostos
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Extrair o rosto da imagem
                x, y = max(0, x), max(0, y)
                w, h = min(w, iw - x), min(h, ih - y)
                face = image[y:y+h, x:x+w]

                if face.size > 0:
                    try:
                        # Pré-processar o rosto detectado
                        face_preprocessed = preprocess_face(face)
                        if face_preprocessed is None:
                            continue

                        # Fazer a previsão de máscara/no máscara
                        prediction = model.predict(face_preprocessed, verbose=0)
                        mask_prob = prediction[0][0]
                        label = "Mask On" if mask_prob > 0.5 else "Mask Off"
                        color = (0, 255, 0) if mask_prob > 0.5 else (255, 0, 0)

                        # Desenhar a caixa delimitadora e a label
                        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(image_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except Exception as e:
                        print(f"Erro ao processar o rosto: {e}")

    # Converter a imagem para exibir na interface Tkinter
    img_pil = Image.fromarray(image_rgb)
    img_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)

    img_tk = ImageTk.PhotoImage(image=img_pil)

    image_label.imgtk = img_tk
    image_label.config(image=img_tk)

# Função de detecção de máscara na webcam com threading seguro
def run_webcam_safe(video_label):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a webcam.")
            cap.release()
            return

        try:
            # Convertendo a imagem para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar o quadro para detecção de rosto
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)

                    # Garantir que as coordenadas estão dentro da imagem
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, iw - x), min(h, ih - y)

                    # Extrair o rosto
                    face = frame[y:y+h, x:x+w]

                    if face.size > 0:
                        # Pré-processar o rosto e fazer a previsão
                        face_preprocessed = preprocess_face(face)
                        if face_preprocessed is None:
                            continue

                        prediction = model.predict(face_preprocessed, verbose=0)
                        mask_prob = prediction[0][0]
                        label = "Mask On" if mask_prob > 0.5 else "Mask Off"
                        color = (0, 255, 0) if mask_prob > 0.5 else (0, 0, 255)

                        # Desenhar a caixa delimitadora e a label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            print(f"Erro durante a detecção na webcam: {e}")

        # Atualizar a imagem na interface Tkinter
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_pil = Image.fromarray(frame_bgr)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        video_label.imgtk = frame_tk
        video_label.config(image=frame_tk)

        # Atualizar o frame após 10ms
        video_label.after(10, update_frame)

    update_frame()

# Classe da aplicação de detecção de máscaras
class MaskDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Detector")
        self.root.geometry("800x600")
        
        self.create_buttons()

    def choose_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            self.show_image(image_path)

    def show_image(self, image_path):
        self.clear_screen()

        image_label = tk.Label(self.root)
        image_label.pack()

        threading.Thread(target=process_image, args=(image_path, image_label)).start()

        start_button = tk.Button(self.root, text="Início", command=self.restart)
        start_button.pack(pady=20)
        
    def create_buttons(self):
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=20)
        
        self.webcam_button = tk.Button(self.button_frame, text="Mask Detector (Webcam)", command=self.start_webcam)
        self.webcam_button.pack(side="left", padx=20)
        
        self.image_button = tk.Button(self.button_frame, text="Mask Detector (Image)", command=self.choose_image)
        self.image_button.pack(side="right", padx=20)
        
    def start_webcam(self):
        self.clear_screen()
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        self.start_button = tk.Button(self.root, text="Início", command=self.restart)
        self.start_button.pack(pady=20)
        
        threading.Thread(target=run_webcam_safe, args=(self.video_label,)).start()
    
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def restart(self):
        self.clear_screen()
        self.create_buttons()

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectorApp(root)
    root.mainloop()
