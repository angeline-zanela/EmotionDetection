#Essa interface foi construída para Classificar emoções em LIBRAS, sendo assim será aberto a camera em tempo real e capturado uma imagem para ser classificada.

import streamlit as st
import dlib
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.applications import VGG16
from PIL import Image

# Foi usado o classificador Haar Cascade pré-treinado para detecção de rostos.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Usado esse arquivo para carregar os pontos do rosto
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Usado o mediapipe para reconhecer os pontos das mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Cria o modelo, mas ele já está treinado
def create_face_model(input_shape_image, input_shape_landmarks, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    base_model.trainable = True

    image_input = Input(shape=input_shape_image)
    landmarks_input = Input(shape=input_shape_landmarks)

    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    combined = Concatenate()([x, landmarks_input])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[image_input, landmarks_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Essa função detecta on pontos com a dlib
def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Pega apenas 1 rosto na imagem
        face_region = image[y:y+h, x:x+w]  # Recorta a região do rosto
        face_region_gray = gray[y:y+h, x:x+w]  # Região do rosto em escala de cinza para melhorar a classificação
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(face_region_gray, rect)
        landmark_points = []
        for i in range(0, 68):  # Usa os 68 landmarks fornecidos pelo dlib no .dat
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return face_region, np.array(landmark_points).flatten()
    else:
        return None, None  # Retorna nada se nenhum rosto for detectado

# Função que verifica a validade dos landmarks
def are_landmarks_valid(landmarks, image_shape):
    if landmarks is None or len(landmarks) != 68 * 2:
        return False
    
    x_points = landmarks[0::2]
    y_points = landmarks[1::2]

    if max(x_points) - min(x_points) < 0.2 * image_shape[1] or max(y_points) - min(y_points) < 0.2 * image_shape[0]:
        return False

    return True

# Função que extrai os pontos da mão
def extract_hand_landmarks(img):
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_points = []
        for lm in hand_landmarks.landmark:
            landmark_points.extend([lm.x, lm.y, lm.z])
        return np.array(landmark_points)
    else:
        return None

# Essa função tem o papel de combinar as predições dos dois modelos para poder fazer a classificação conjunta do modelo da face e da mão
def combine_predictions(face_pred, gesture_pred):
    # Combine as predições utilizando a média, sendo que se aparecer apenas a face seu máximo será de 50% e se aparecer apenas o gesto seu máximo tb será de 50%
    combined_confidences = (face_pred + gesture_pred) / 2
    dominant_class = np.argmax(combined_confidences)
    confidence = combined_confidences[dominant_class] * 100  # Confiança final como porcentagem
    return dominant_class, confidence

# Carregar os modelos e seus pesos, sendo bem importante e isso é definido no código de treinamento
num_classes = 4  # 'Feliz', 'Raiva', 'Surpreso', 'Triste'
face_model = create_face_model((224, 224, 3), (68 * 2,), num_classes)
# Atualize se quiser testar cos outros 2 modelos treinados da face em diferentes contextos, o que está setado é o que tem maior taxa de sucesso
face_model.load_weights('modelo_emocao_face_4classes.weights.h5')

Esse modelo foi o mais assertivo então ele foi o único gerado
gesture_model = tf.keras.models.load_model('modelo_landmarks_gesto_emocoes_libras.h5') 

# Função usada para capturar a imagem da câmera do navegador usando Streamlit, será tirado uma imagem em tempo real e classificado em seguida
def classify_realtime_streamlit():
    st.title("Detecção de Emoções e Gestos em Tempo Real")
    uploaded_image = st.camera_input("Capture uma imagem usando a câmera")

    if uploaded_image is not None:
        # Converte a imagem capturada em um array
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption='Imagem capturada', use_column_width=True)

        class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']

        # Detectar emoção facial
        face_region, face_landmarks = detect_face_and_landmarks(image)
        face_prediction = np.zeros((1, num_classes))  # Previsão padrão

        if face_region is not None and face_landmarks is not None and are_landmarks_valid(face_landmarks, image.shape):
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_region_resized = cv2.resize(face_region_rgb, (224, 224))
            face_region_resized = np.expand_dims(face_region_resized.astype(np.float32) / 255.0, axis=0)
            face_landmarks = np.expand_dims(face_landmarks, axis=0)

            face_prediction = face_model.predict([face_region_resized, face_landmarks])

        # Detecta os gestos da mão
        hand_landmarks = extract_hand_landmarks(image)
        gesture_prediction = np.zeros((1, num_classes)) 

        if hand_landmarks is not None:
            hand_landmarks = np.expand_dims(hand_landmarks, axis=0)
            gesture_prediction = gesture_model.predict(hand_landmarks)

        # Combina as predições e determinar a classe dominante
        dominant_class, confidence = combine_predictions(face_prediction[0], gesture_prediction[0])

        # Exibe o resultado na tela
        label = f"{class_names[dominant_class]}: {confidence:.2f}%"
        st.write(f"Emoção detectada: {label}")

# Inicia a aplicação Streamlit
if __name__ == '__main__':
    classify_realtime_streamlit()
