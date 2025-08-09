from __future__ import annotations

import cv2
import numpy as np


def _deskew(image: np.ndarray) -> np.ndarray:

    coords = np.column_stack(np.where(image > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def clean_and_binarize(image: np.ndarray) -> np.ndarray:
    # Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Filtro bilateral preserva bordes mientras reduce ruido
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Mejora de contraste para fotos con poca luz
    gray = cv2.equalizeHist(gray)

    # Binarizacion adaptativa para obtener texto en blanco sobre fondo negro
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )

    # Operaciones morfologicas para eliminar ruido y cerrar huecos
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Corregir inclinacion
    bw = _deskew(bw)
    # Reaplicar un umbral para asegurar binarizacion tras la rotacion
    _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw