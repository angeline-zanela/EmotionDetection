import dlib
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.applications import VGG16

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_face_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0] 
        face_region = image[y:y+h, x:x+w] 
        face_region_gray = gray[y:y+h, x:x+w]
        rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(face_region_gray, rect)
        landmark_points = []
        for i in range(0, 68):  
            landmark_points.append([landmarks.part(i).x, landmarks.part(i).y])
        return face_region, np.array(landmark_points).flatten()
    else:
        return None, None 

def are_landmarks_valid(landmarks, image_shape):
    if landmarks is None or len(landmarks) != 68 * 2:
        return False
    x_points = landmarks[0::2]
    y_points = landmarks[1::2]
    if max(x_points) - min(x_points) < 0.2 * image_shape[1] or max(y_points) - min(y_points) < 0.2 * image_shape[0]:
        return False
    return True
  
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

num_classes = 4  # 'Feliz', 'Raiva', 'Surpreso', 'Triste'
face_model = create_face_model((224, 224, 3), (68 * 2,), num_classes)
face_model.load_weights('modelo_emocao_face__V3_4classes.weights.h5')
gesture_model = tf.keras.models.load_model('modelo_landmarks_gesto_emocoes_libras.h5') 

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

def combine_predictions(face_pred, gesture_pred):
    combined_confidences = (face_pred + gesture_pred) / 2
    dominant_class = np.argmax(combined_confidences)
    confidence = combined_confidences[dominant_class] * 100 
    return dominant_class, confidence

def classify_realtime():
    frame_count = 0
    skip_frames = 2 
    class_names = ['Feliz', 'Raiva', 'Surpreso', 'Triste']
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Falha ao capturar imagem da câmera.")
            break
        if frame_count % skip_frames != 0:
            continue 
        face_region, face_landmarks = detect_face_and_landmarks(frame)
        face_prediction = np.zeros((1, num_classes))  
        if face_region is not None and face_landmarks is not None and are_landmarks_valid(face_landmarks, frame.shape):
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_region_resized = cv2.resize(face_region_rgb, (224, 224))
            face_region_resized = np.expand_dims(face_region_resized.astype(np.float32) / 255.0, axis=0)
            face_landmarks = np.expand_dims(face_landmarks, axis=0)

            face_prediction = face_model.predict([face_region_resized, face_landmarks])
        hand_landmarks = extract_hand_landmarks(frame)
        gesture_prediction = np.zeros((1, num_classes))  
        if hand_landmarks is not None:
            hand_landmarks = np.expand_dims(hand_landmarks, axis=0)
            gesture_prediction = gesture_model.predict(hand_landmarks)
        dominant_class, confidence = combine_predictions(face_prediction[0], gesture_prediction[0])
        label = f"{class_names[dominant_class]}: {confidence:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emoção e Gesto em Tempo Real', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Executa a função de classificação em tempo real
classify_realtime()
