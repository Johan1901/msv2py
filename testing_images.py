from src.image_processor import process_equation_image, LABEL_MAP
import cv2

image_path = "test_images/ez.png"

img, recognitions, equation, verdict = process_equation_image(image_path)
print(recognitions)  # [(x, y, w, h, label, prob), ...]

for (x, y, w, h, label, _prob) in recognitions:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, LABEL_MAP.get(label, label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("resultado_prueba.jpg", img)
