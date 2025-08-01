import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

# Constantes
MODEL_PATH = 'model/math_symbol_recognizer.h5'
LABELS_PATH = 'model/class_labels.json'
IMG_WIDTH, IMG_HEIGHT = 28, 28

def process_equation_image(image_path):
    # 1. Cargar modelo y etiquetas
    try:
        model = load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        labels_to_class = {v: k for k, v in class_labels.items()}
    except IOError as e:
        print(f"Error al cargar el modelo o las etiquetas: {e}")
        return None, []

    # 2. Cargar y pre-procesar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo leer la imagen en {image_path}")
        return None, []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # El umbral adaptativo sigue siendo la mejor opción general
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # 3. Limpieza morfológica (MENOS AGRESIVA)
    # Esto ayuda a conectar partes rotas de un caracter sin eliminarlo por completo
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    # 4. Encontrar contornos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filtrar contornos con un método FIJO pero ROBUSTO
    bounding_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # Filtro de tamaño y relación de aspecto para aceptar solo caracteres probables
        # Se ajustan los valores para ser un poco más permisivos que antes
        if (w >= 10 and h >= 15) and (w < 2.5 * h):
            bounding_boxes.append((x, y, w, h))
            
    # 6. Ordenar y predecir
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    recognitions = []
    for (x, y, w, h) in bounding_boxes:
        roi = thresh[y:y+h, x:x+w]
        padded_roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_roi = cv2.resize(padded_roi, (IMG_WIDTH, IMG_HEIGHT))
        processed_roi = resized_roi.astype("float32") / 255.0
        processed_roi = np.expand_dims(processed_roi, axis=-1)
        processed_roi = np.expand_dims(processed_roi, axis=0)
        
        prediction = model.predict(processed_roi, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = labels_to_class[predicted_index]
        
        recognitions.append((x, y, w, h, predicted_label))

    return image, recognitions