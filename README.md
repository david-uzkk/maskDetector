# Mask Detector

Este projeto realiza detecção de máscaras faciais usando TensorFlow, OpenCV, MediaPipe e Tkinter. Permite detecção em tempo real via webcam ou em imagens estáticas.

---

## Requisitos
- Python 3.8+
- Dependências no arquivo `requirements.txt`

## Uso

python main.py

## Estrutura do Projeto

- main.py: Interface Tkinter.
- mask_detector.py: Detecção via webcam.
- mask_detector_img.py: Detecção em imagens.

## Modelo

- O modelo keras_model.h5 foi treinado para classificar rostos como "com máscara" ou "sem máscara".