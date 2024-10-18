#################################################################
###  CRIADO PARA TESTAR OS MODELOS EM TEMPO REAL COM A CAMERA ###
#################################################################


import numpy as np
import tensorflow as tf
import cv2
import dlib
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout

# Essa função cria um modelo de rede neural para reconhecimento facial,
# que combina a imagem do rosto e landmarks (pontos faciais) para melhorar a classificação de emoções.
def create_face_model(input_shape_image, input_shape_landmarks, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_image)
    base_model.trainable = True

    image_input = Input(shape=input_shape_image)
    landmarks_input = Input(shape=input_shape_landmarks)

    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Regularização para evitar overfitting

    combined = Concatenate()([x, landmarks_input])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)  # Mais regularização para maior generalização
    output = Dense(num_classes, activation='softmax')(combined)  # A camada de saída classifica as emoções

    model = tf.keras.Model(inputs=[image_input, landmarks_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Aqui é recriado o modelo facial. Como no modelo da face o treinamento só veio os pesos, precisa ser definido a arquitetura.
# O modelo classifica emoções em 4 categorias: 'Feliz', 'Raiva', 'Surpresa', 'Triste'.
num_classes = 4  
face_model = create_face_model((224, 224, 3), (68 * 2,), num_classes)

# Carrega os pesos previamente treinados para o modelo facial.
face_model.load_weights('modelo_emocao_face_4classes.weights.h5')

# Carrega o modelo de gestos para a detecção de emoções com base nos movimentos das mãos.
gesture_model = tf.keras.models.load_model('modelo_landmarks_gesto_emocoes_libras.h5')

# Inicializa MediaPipe, uma biblioteca usada para detectar landmarks das mãos.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializa o preditor de landmarks faciais usando o `dlib`.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Essa função detecta rostos em uma imagem usando Haar Cascade.
# Extrai landmarks faciais (68 pontos) usando o `dlib` para melhorar a precisão na classificação.
# A imagem do rosto é redimensionada para 224x224, compatível com o modelo VGG16.
def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]  
        face_region = image[y:y+h, x:x+w]
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, rect)
        landmark_points = []
        for i in range(68):  # Os 68 pontos dos landmarks faciais
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return cv2.resize(face_region, (224, 224)), np.array(landmark_points).flatten()
    return np.zeros((224, 224, 3)), np.zeros(68 * 2)  # Caso nenhum rosto seja detectado, retorna valores vazios


# Essa função usa MediaPipe para detectar landmarks das mãos em uma imagem.
# Retorna as coordenadas dos landmarks (pontos da mão).
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


# Essa função recebe a imagem facial e landmarks faciais e de mão.
# Faz a predição combinada das emoções a partir do modelo facial e do modelo de gestos.
# Combina as probabilidades de ambos os modelos e retorna a classe final (emoção) e as probabilidades combinadas.
def combine_predictions(face_image, face_landmarks, hand_landmarks):
    face_probs = face_model.predict([np.expand_dims(face_image, axis=0), np.expand_dims(face_landmarks, axis=0)])
    gesture_probs = gesture_model.predict(np.expand_dims(hand_landmarks, axis=0))
    combined_probs = (face_probs + gesture_probs) / 2.0  # Média das probabilidades dos dois modelos
    final_class = np.argmax(combined_probs, axis=1)  # Predição final: a classe com maior probabilidade
    return final_class, combined_probs

# Desenha o nome da classe e a precisão na imagem de entrada.
# Usa o OpenCV para exibir o texto na imagem.
def draw_text_on_image(image, text, position=(30, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Essa função captura frames da câmera em tempo real.
# Faz a predição das emoções combinando os modelos de face e gestos.
# Atualiza a exibição da imagem e da classificação com Matplotlib em tempo real.
def classify_emotion_real_time():
    cap = cv2.VideoCapture(0)
    class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']
    plt.ion() 
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Classificação de emoções em LIBRAS em tempo real', fontsize=16)
    img_plot = None
    text = "Detecção de emoção"  # Texto inicial
    # Loop para capturar e processar frames da câmera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da câmera")
            break
        
        original_image = frame.copy()
        
        # Detecta rosto e landmarks faciais
        face_image_resized, face_landmarks = detect_face_and_landmarks(frame)
        
        # Detecta landmarks das mãos
        hand_landmarks = extract_hand_landmarks(frame)
        
        if hand_landmarks is not None and face_landmarks is not None:
            # Predição da emoção com os dois modelos
            emotion_class, probabilities = combine_predictions(face_image_resized, face_landmarks, hand_landmarks)
            
            # Define o nome da classe e a precisão
            emotion_label = class_names[int(emotion_class)]
            precision = np.max(probabilities)
            text = f"{emotion_label}: {precision * 100:.2f}%"
            
            # Desenha o texto na imagem original
            draw_text_on_image(original_image, text)
        
        # Atualiza o gráfico em tempo real com Matplotlib
        if img_plot is None:
            img_plot = ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            img_plot.set_data(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        ax.axis('off')  # Remove os eixos da imagem
        plt.pause(0.001)  # Atualiza a imagem a cada iteração
    
    cap.release()  # Libera a câmera após o uso
    plt.ioff() 
    plt.show()

classify_emotion_real_time()
