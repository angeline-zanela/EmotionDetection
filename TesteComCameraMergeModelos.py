#################################################################
###  CRIADO PARA TESTAR OS MODELOS EM TEMPO REAL COM A CAMERA ###
#################################################################


import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import dlib
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from IPython.display import Javascript, display
from google.colab.output import eval_js
from base64 import b64decode

class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']

def combine_predictions(face_probs, gesture_probs):
    combined_probs = (face_probs + gesture_probs) / 2
    final_class = np.argmax(combined_probs)
    confidence = combined_probs[final_class] * 100
    return final_class, confidence

def load_gesture_model(model_path):
    return tf.keras.models.load_model(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("/content/Modelos/shape_predictor_68_face_landmarks.dat")

def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, rect)
        landmark_points = []
        for i in range(68):
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return cv2.resize(face_region, (224, 224)), np.array(landmark_points).flatten()
    return None, None

# Função para capturar imagem da câmera via JavaScript
def take_photo(filename='photo.jpg', quality=0.8):
    js = """
    async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = 'Tire sua Foto';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getTracks().forEach(track => track.stop());
        div.remove();

        return canvas.toDataURL('image/jpeg', quality);
    }
    takePhoto({quality});
    """
    display(Javascript(js))
    data = eval_js("takePhoto({})".format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

def classify_combined_model(image_path, face_model, gesture_model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem no caminho '{image_path}'.")
        return

    face_image, face_landmarks = detect_face_and_landmarks(image)
    hand_landmarks = extract_hand_landmarks(image)

    if face_image is None or face_landmarks is None or hand_landmarks is None:
        print("Face ou gestos não detectados.")
        return
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image_resized = np.expand_dims(face_image_rgb.astype(np.float32) / 255.0, axis=0)
    face_landmarks = np.expand_dims(face_landmarks, axis=0)
    face_probs = face_model.predict([face_image_resized, face_landmarks])
    hand_landmarks = np.expand_dims(hand_landmarks, axis=0)
    gesture_probs = gesture_model.predict(hand_landmarks)
    final_class, confidence = combine_predictions(face_probs[0], gesture_probs[0])
    label = f"{class_names[final_class]} ({confidence:.2f}%)"
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(label)
    plt.axis('off')
    plt.show()

gesture_model = load_gesture_model('/content/Modelos/modelo_landmarks_gesto_emocoes_libras.h5')
face_model = create_face_model((224, 224, 3), (68 * 2,), len(class_names))
face_model.load_weights('/content/Modelos/modelo_emocao_face_4classes.weights.h5')
photo_filename = take_photo()
classify_combined_model(photo_filename, face_model, gesture_model)
    plt.show()

classify_emotion_real_time()
