## app.py
import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
from pymongo import MongoClient
from src.image_processor import process_equation_image
from src.main import evaluate_and_verify

# ---------------------- CONFIGURACIÓN ----------------------

# Flask
app = Flask(__name__)

# MongoDB Atlas
MONGO_URI = "mongodb+srv://felipeburgos1901:12345@cluster0.vvhuunc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["gatolate"]
collection = db["resultados_ejercicios"]

# Carpeta temporal para imágenes
TEMP_IMG_PATH = "temp/resultado.jpg"
os.makedirs("temp", exist_ok=True)

# ---------------------- ENDPOINT: /procesar ----------------------

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se envío ninguna imagen"}), 400

    archivo = request.files['imagen']
    image_bytes = archivo.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if imagen is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400

    temp_input_path = os.path.join("temp", archivo.filename)
    cv2.imwrite(temp_input_path, imagen)

    original_image, recognitions = process_equation_image(temp_input_path)
    if original_image is None:
        return jsonify({"error": "Error al procesar imagen"}), 500

    LABEL_MAP = {"add": "+", "sub": "-", "eq": "="}
    equation_str = ""
    for (x, y, w, h, label) in recognitions:
        symbol = LABEL_MAP.get(label, label)
        equation_str += symbol
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, symbol, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    veredicto, _ = evaluate_and_verify(equation_str)

    h, w, _ = original_image.shape
    color = (0, 255, 0) if "Correcto" in veredicto else (0, 0, 255)
    cv2.putText(original_image, f"Veredicto: {veredicto}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(TEMP_IMG_PATH, original_image)

    _, buffer = cv2.imencode('.jpg', original_image)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "ecuacion": equation_str,
        "veredicto": veredicto,
        "imagen_base64": base64_img
    })

# ---------------------- ENDPOINT: /guardar ----------------------

@app.route('/guardar', methods=['POST'])
def guardar_resultado():
    data = request.get_json()

    alumno = data.get("alumno")
    ecuacion = data.get("ecuacion")
    veredicto = data.get("veredicto")
    imagen_base64 = data.get("imagen_base64")

    if not all([alumno, ecuacion, veredicto, imagen_base64]):
        return jsonify({"error": "Faltan datos"}), 400

    documento = {
        "alumno": alumno,
        "fecha": datetime.utcnow(),
        "ecuacion": ecuacion,
        "veredicto": veredicto,
        "imagen": imagen_base64
    }

    resultado = collection.insert_one(documento)
    return jsonify({"mensaje": "Guardado correctamente", "id": str(resultado.inserted_id)})

# ---------------------- INICIO ----------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)


## main.py (solo necesitas corregir el import y asegurar que se pueda importar desde src)

import cv2
from src.image_processor import process_equation_image

LABEL_MAP = {
    "add": "+",
    "sub": "-",
    "eq": "="
}

def evaluate_and_verify(equation_str):
    try:
        if '=' not in equation_str:
            return "Invalida (sin '=')", None
        parts = equation_str.split('=')
        operation_part = parts[0]
        expected_result_str = parts[1]
        calculated_result = eval(operation_part)
        expected_result = int(expected_result_str)
        if calculated_result == expected_result:
            return "Correcto", calculated_result
        else:
            return f"Incorrecto, deberia ser {calculated_result}", calculated_result
    except (SyntaxError, ValueError, ZeroDivisionError, IndexError):
        return "Expresión mal formada", None

def main():
    import os
    image_path = os.path.join('test_images', 'uys.png')
    output_path = 'resultado_con_cajas.jpg'

    original_image, recognitions = process_equation_image(image_path)

    if original_image is None:
        return

    equation_str = ""
    for (x, y, w, h, label) in recognitions:
        display_symbol = LABEL_MAP.get(label, label)
        equation_str += display_symbol
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, display_symbol, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"Ecuación reconocida: {equation_str}")

    verdict = "Sin resultado"
    verdict_color = (0, 0, 255)

    if equation_str:
        verdict, result = evaluate_and_verify(equation_str)
        print(f"Veredicto: {verdict}")
        if "Correcto" in verdict:
            verdict_color = (0, 255, 0)

    h, w, _ = original_image.shape
    cv2.putText(original_image, f"Veredicto: {verdict}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)

    cv2.imwrite(output_path, original_image)
    print(f"Imagen con resultado guardada en: {output_path}")
    cv2.imshow("Resultado del Reconocimiento", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()