from __future__ import annotations
import base64
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .image_processor import process_equation_image


LABEL_MAP = {"add": "+", "sub": "-", "eq": "="}


def evaluate_and_verify(equation_str: str) -> Tuple[str, Optional[int]]:

    try:
        if "=" not in equation_str:
            return "Invalida (sin '=')", None

        operation_part, expected_result_str = equation_str.split("=")
        calculated_result = eval(operation_part)
        expected_result = int(expected_result_str)

        if calculated_result == expected_result:
            return "Correcto", calculated_result

        return f"Incorrecto, deberia ser {calculated_result}", calculated_result

    except (SyntaxError, ValueError, ZeroDivisionError, IndexError):
        return "Expresión mal formada", None


def process_and_store_image(
    image_bytes: bytes, alumno: Optional[str], collection
) -> Tuple[Dict[str, str], Optional[str]]:

    np_arr = np.frombuffer(image_bytes, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if imagen is None:
        return {}, "No se pudo decodificar la imagen"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name

    try:
        original_image, recognitions = process_equation_image(temp_path)
    finally:
        os.remove(temp_path)

    if original_image is None:
        return {}, "Error al procesar imagen"

    equation_str = ""
    for (x, y, w, h, label) in recognitions:
        symbol = LABEL_MAP.get(label, label)
        equation_str += symbol
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            symbol,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    veredicto, _ = evaluate_and_verify(equation_str)

    h, w, _ = original_image.shape
    color = (0, 255, 0) if "Correcto" in veredicto else (0, 0, 255)
    cv2.putText(
        original_image,
        f"Veredicto: {veredicto}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    _, buffer = cv2.imencode(".jpg", original_image)
    base64_img = base64.b64encode(buffer).decode("utf-8")

    documento = {
        "alumno": alumno,
        "fecha": datetime.utcnow(),
        "ecuacion": equation_str,
        "veredicto": veredicto,
        "imagen": base64_img,
    }
    result = collection.insert_one(documento)

    return (
        {
            "ecuacion": equation_str,
            "veredicto": veredicto,
            "imagen_base64": base64_img,
            "id": str(result.inserted_id),
        },
        None,
    )
