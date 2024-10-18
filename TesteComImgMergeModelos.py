############################################################
###  CRIADO PARA TESTAR OS MODELOSCOM IMAGENS DE ENTRADA ###
############################################################

#Blibiotecas necessárias 
import numpy as np
import tensorflow as tf
import cv2
import dlib
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout

# A VGG16: A rede VGG16 é usada como base para extrair características da imagem facial.
# Os Landmarks Faciais: São adicionadas informações dos pontos faciais como uma segunda entrada para melhorar a precisão.
# Arquitetura: Camadas adicionais são conectadas para combinar a saída da VGG16 com os pontos dos landmarks, 
# e uma camada de saída usa softmax(função matemática, converte os valores de saída (logits) 
# de uma rede neural em probabilidades, normalizando, de modo que cada valor represente uma 
# probabilidade associada a uma classe) e assim poder classificar as emoções.

# Nessa função foi preciso reconstruir o modelo treinado por motivos de o modelo apenas conter 
# os pesos e não a arquitetura do treinamento, precisando recriar o modelo antes de carregar os pesos.

def create_face_model(input_shape_image, input_shape_landmarks, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    base_model.trainable = True

    image_input = Input(shape=input_shape_image)
    landmarks_input = Input(shape=input_shape_landmarks)

    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Regularização com Dropout

    combined = Concatenate()([x, landmarks_input])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)  # Regularização na camada combinada
    output = Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[image_input, landmarks_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Recria o modelo facial
num_classes = 4  # 'Feliz', 'Raiva', 'Surpresa', 'Tristeza'
face_model = create_face_model((224, 224, 3), (68 * 2,), num_classes)

# Carrega os pesos do modelo da face
face_model.load_weights('modelo_emocao_face_4classes.weights.h5')

# Carrega o modelo Gestual(Nesse não foi preciso reconstruir pois ele foi treinado com a arquitetura e os pesos corretamente)
gesture_model = tf.keras.models.load_model('modelo_landmarks_gesto_emocoes_libras.h5') 

# Inicializa o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Carrega o preditor de landmarks faciais da dlib
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Função para detectar rosto e landmarks faciais 
#O Haar Cascade: Detecta o rosto na imagem.
#A dlib: Extrai 68 pontos de landmarks do rosto detectado.
# O Redimensionamento da Imagem: O rosto detectado é redimensionado para 224x224 pixels, necessário para o modelo VGG16.
def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]  
        face_region = image[y:y+h, x:x+w]
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, rect)
        landmark_points = []
        for i in range(68):  # 68 landmarks fornecidos pelo dlib
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return cv2.resize(face_region, (224, 224)), np.array(landmark_points).flatten()
    return np.zeros((224, 224, 3)), np.zeros(68 * 2)  # Retorna uma imagem em branco e landmarks vazios

# Função para extrair pontos da mão
def extract_hand_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_points = []
        for lm in hand_landmarks.landmark:
            landmark_points.extend([lm.x, lm.y, lm.z])
        return np.array(landmark_points)
    return None

# Função para combinar predições de ambos os modelos
# Predição Facial: Usa o modelo facial para obter a probabilidade de cada classe com base na imagem do rosto e nos landmarks faciais.
# Predição Gestual: Usa o modelo de gestos para obter a probabilidade de cada classe com base nos landmarks das mãos.
# Combinação de Probabilidades: As probabilidades de ambos os modelos são combinadas usando uma média simples.
def combine_predictions(face_image, face_landmarks, hand_landmarks):
    face_probs = face_model.predict([np.expand_dims(face_image, axis=0), np.expand_dims(face_landmarks, axis=0)])
    gesture_probs = gesture_model.predict(np.expand_dims(hand_landmarks, axis=0))
    combined_probs = (face_probs + gesture_probs) / 2.0
    final_class = np.argmax(combined_probs, axis=1)
    return final_class, combined_probs

# Função para mostrar a classificação na imagem
def draw_text_on_image(image, text, position=(30, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Função para carregar a imagem, realizar a predição e exibir a imagem com resultados
def classify_emotion(image_path):
    # Carrega e processa a imagem
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # Extrai landmarks faciais e região da face
    face_image_resized, face_landmarks = detect_face_and_landmarks(image)
    
    # Extrai landmarks das mãos
    hand_landmarks = extract_hand_landmarks(image)
    
    # Define os nomes das classes
    class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']

    if hand_landmarks is not None and face_landmarks is not None:
        # Combina as predições dos dois modelos
        emotion_class, probabilities = combine_predictions(face_image_resized, face_landmarks, hand_landmarks)
        
        # Obtem a classe e a precisão
        emotion_label = class_names[int(emotion_class)]
        precision = np.max(probabilities)
        text = f"{emotion_label}: {precision * 100:.2f}%"
        draw_text_on_image(original_image, text)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(text)
        plt.axis('off')  
        plt.show()
    else:
        print("Não foi possível detectar a face ou a mão adequadamente.")

# Exemplo de uso
classify_emotion('img/x.jpg')
