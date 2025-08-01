import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Constantes y Configuración ---
DATASET_PATH = 'model/dataset'
MODEL_SAVE_PATH = 'model/math_symbol_recognizer.h5'
LABELS_SAVE_PATH = 'model/class_labels.json'
IMG_WIDTH, IMG_HEIGHT = 28, 28
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 15
BATCH_SIZE = 32

# --- Clases que vamos a entrenar (AQUÍ ESTÁ EL CAMBIO) ---
# Se agregó 'eq' a la lista para reconocer el signo de igual
CLASSES_TO_TRAIN = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'sub', 'eq']

def load_dataset(dataset_path, classes, img_width, img_height):
    """
    Carga imágenes desde las carpetas del dataset, las preprocesa y las etiqueta.
    """
    images = []
    labels = []
    
    # Crear un mapa de nombre de clase a un número (etiqueta)
    # Ej: {'0': 0, ..., 'add': 10, 'sub': 11, 'eq': 12}
    class_to_label = {class_name: i for i, class_name in enumerate(classes)}
    
    print("Cargando dataset...")
    print(f"Clases a cargar: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):
            print(f"¡Advertencia! La carpeta no existe: {class_path}")
            continue
            
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Cargando {len(image_files)} imágenes de la clase '{class_name}'...")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            # Cargar imagen en escala de grises
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"No se pudo leer la imagen: {img_path}")
                continue
            
            # Invertir imagen si es necesario (Keras espera fondo negro, objeto blanco)
            img = cv2.bitwise_not(img)
            
            # Redimensionar y normalizar
            img = cv2.resize(img, (img_width, img_height))
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(class_to_label[class_name])

    if not images:
        raise ValueError("No se cargaron imágenes. Verifica la ruta del dataset y las carpetas de clases.")

    # Convertir a numpy arrays y añadir una dimensión para el canal de color (escala de grises)
    images = np.array(images).reshape(-1, img_width, img_height, 1)
    labels = np.array(labels)
    
    # Guardar el mapa de etiquetas para usarlo en la predicción
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_to_label, f)
    print(f"Mapa de etiquetas guardado en {LABELS_SAVE_PATH}")
    
    return images, labels, len(classes)

def build_model(input_shape, num_classes):
    """
    Construye, compila y retorna el modelo de Red Neuronal Convolucional.
    """
    model = Sequential([
        # Capa de convolución 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Capa de convolución 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Aplanar la salida para la capa Densa
        Flatten(),
        
        # Capa Densa y Dropout para evitar sobreajuste
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Capa de salida con activación 'softmax' para clasificación
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- Flujo Principal de Ejecución ---
if __name__ == '__main__':
    # 1. Cargar y preparar los datos
    images, labels, num_classes = load_dataset(DATASET_PATH, CLASSES_TO_TRAIN, IMG_WIDTH, IMG_HEIGHT)
    
    # Convertir etiquetas a formato one-hot encoding (necesario para 'categorical_crossentropy')
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    # 2. Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_categorical, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=labels_categorical
    )
    
    print(f"\nTotal de imágenes para entrenamiento: {X_train.shape[0]}")
    print(f"Total de imágenes para validación: {X_test.shape[0]}")
    
    # 3. Construir el modelo
    model = build_model((IMG_WIDTH, IMG_HEIGHT, 1), num_classes)
    model.summary() # Verás que la capa de salida ahora tiene 13 neuronas en lugar de 12
    
    # 4. Entrenar el modelo
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    
    # 5. Guardar el modelo entrenado
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ ¡Entrenamiento completado! Modelo guardado en: {MODEL_SAVE_PATH}")