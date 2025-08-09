import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from .preprocessing import clean_and_binarize

# Constantes
MODEL_PATH = "model/math_symbol_recognizer.h5"
LABELS_PATH = "model/class_labels.json"
IMG_WIDTH, IMG_HEIGHT = 28, 28

# Símbolos legibles y umbral de confianza mínimo
LABEL_MAP = {"add": "+", "sub": "-", "eq": "="}
CONFIDENCE_THRESHOLD = 0.8

def _merge_equal_sign_boxes(boxes, gap_threshold=20, width_diff=15, x_diff=15):

    merged = []
    skip = set()

    for i in range(len(boxes)):
        if i in skip:
            continue
        x1, y1, w1, h1 = boxes[i]
        new_box = None

        for j in range(i + 1, len(boxes)):
            if j in skip:
                continue
            x2, y2, w2, h2 = boxes[j]

            if abs(x1 - x2) <= x_diff and abs(w1 - w2) <= width_diff:
                if y1 < y2:
                    gap = y2 - (y1 + h1)
                else:
                    gap = y1 - (y2 + h2)

                if 0 <= gap <= gap_threshold:
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    new_box = (x, y, w, h)
                    skip.add(j)
                    break

        if new_box:
            merged.append(new_box)
            skip.add(i)
        else:
            merged.append((x1, y1, w1, h1))

    return merged


def _dedupe_overlapping_boxes(boxes, iou_threshold=0.8):

    def _iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union else 0

    deduped = []
    for box in boxes:
        merged = False
        for idx, existing in enumerate(deduped):
            if _iou(box, existing) > iou_threshold:
                x, y, w, h = box
                ex, ey, ew, eh = existing
                nx = min(x, ex)
                ny = min(y, ey)
                nw = max(x + w, ex + ew) - nx
                nh = max(y + h, ey + eh) - ny
                deduped[idx] = (nx, ny, nw, nh)
                merged = True
                break
        if not merged:
            deduped.append(box)
    return deduped

def _evaluate_equation(equation: str) -> str:
    """Evalúa una ecuación simple y devuelve un veredicto."""
    try:
        if "=" not in equation:
            return "Invalida (sin '=')"
        left, right = equation.split("=")
        calculated = eval(left)
        expected = int(right)
        if calculated == expected:
            return "Correcto"
        return f"Incorrecto, deberia ser {calculated}"
    except Exception:
        return "Expresión mal formada"
    

def process_equation_image(image_path):
    # 1. Cargar modelo y etiquetas
    try:
        model = load_model(MODEL_PATH)
        with open(LABELS_PATH, "r") as f:
            class_labels = json.load(f)
        labels_to_class = {v: k for k, v in class_labels.items()}
    except IOError as e:
        print(f"Error al cargar el modelo o las etiquetas: {e}")
        return None, [], "", "Error"

    # 2. Cargar y pre-procesar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo leer la imagen en {image_path}")
        return None, [], "", "Error"
    
    # Aplicar limpieza y binarización robusta para fotografías
    thresh = clean_and_binarize(image)

     # 4. Encontrar contornos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filtrar contornos con un método FIJO pero ROBUSTO
    bounding_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        if area < 50:
            continue
        # Dígitos y símbolos altos
        if w >= 10 and h >= 15 and aspect_ratio < 2.5:
            bounding_boxes.append((x, y, w, h))
        # Símbolos delgados como '-' y '='
        elif w >= 15 and h >= 5 and aspect_ratio >= 2.5:
            bounding_boxes.append((x, y, w, h))

    # 6. Ordenar, combinar y deduplicar posibles signos '=' antes de predecir
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    bounding_boxes = _merge_equal_sign_boxes(bounding_boxes)
    bounding_boxes = _dedupe_overlapping_boxes(bounding_boxes)

    recognitions = []
    equation_str = ""
    for (x, y, w, h) in bounding_boxes:
        roi = thresh[y : y + h, x : x + w]
        padded_roi = cv2.copyMakeBorder(
            roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        resized_roi = cv2.resize(padded_roi, (IMG_WIDTH, IMG_HEIGHT))
        processed_roi = resized_roi.astype("float32") / 255.0
        processed_roi = np.expand_dims(processed_roi, axis=-1)
        processed_roi = np.expand_dims(processed_roi, axis=0)

        prediction = model.predict(processed_roi, verbose=0)[0]
        prob = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))
        predicted_label = labels_to_class[predicted_index]

        # Heurística para evitar que '+' se confunda con '4'
        if predicted_label == "4":
            box_aspect = w / float(h)
            if 0.75 <= box_aspect <= 1.25:
                predicted_label = "add"

        if prob >= CONFIDENCE_THRESHOLD:
            recognitions.append((x, y, w, h, predicted_label, prob))
            equation_str += LABEL_MAP.get(predicted_label, predicted_label)

    veredict = _evaluate_equation(equation_str)
    return image, recognitions, equation_str, veredict