import cv2
from image_processor import process_equation_image

# Mapeo de etiquetas a símbolos legibles
LABEL_MAP = {
    "add": "+",
    "sub": "-",
    "eq": "="
}

def evaluate_and_verify(equation_str):
    # Usar 'eval()' es simple, pero puede ser inseguro en producción.
    # Para este caso controlado, donde solo reconocemos números y operadores, es aceptable.
    try:
        # Separar la ecuación en 'operación' y 'resultado esperado'
        if '=' not in equation_str:
            return "Invalida (sin '=')", None
        
        parts = equation_str.split('=')
        operation_part = parts[0]
        expected_result_str = parts[1]
        
        # Calcular el resultado real de la operación
        calculated_result = eval(operation_part)
        
        # Convertir el resultado esperado a número
        expected_result = int(expected_result_str)
        
        # Comparar y dar un veredicto
        if calculated_result == expected_result:
            return "Correcto", calculated_result
        else:
            return f"Incorrecto, deberia ser {calculated_result}", calculated_result
            
    except (SyntaxError, ValueError, ZeroDivisionError, IndexError):
        # Captura errores como '2++2', '4=', o si la conversión a int falla.
        return "Expresión mal formada", None

def main():
    image_path = 'test_images/ejemplo_ecuacion.jpg'
    #image_path = 'test_images/uys.png'

    output_path = 'resultado_con_cajas.jpg'
    
    original_image, recognitions = process_equation_image(image_path)
    
    if original_image is None:
        return

    equation_str = ""
    
    # Construir el string de la ecuación
    for (x, y, w, h, label) in recognitions:
        display_symbol = LABEL_MAP.get(label, label)
        equation_str += display_symbol
        
        # Dibujar el rectángulo y la etiqueta del caracter
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, display_symbol, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print(f"Ecuación reconocida: {equation_str}")
    
    # --- INICIO DE LA LÓGICA DE VERIFICACIÓN ---
    verdict = "Sin resultado"
    verdict_color = (0, 0, 255) # Rojo por defecto

    if equation_str:
        verdict, result = evaluate_and_verify(equation_str)
        print(f"Veredicto: {verdict}")
        
        if "Correcto" in verdict:
            verdict_color = (0, 255, 0) # Verde
    
    # Escribir el veredicto final en la imagen
    (h, w, _) = original_image.shape
    cv2.putText(original_image, f"Veredicto: {verdict}", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)
    # --- FIN DE LA LÓGICA DE VERIFICACIÓN ---

    # Guardar y mostrar la imagen final
    cv2.namedWindow("Resultado del Reconocimiento", cv2.WINDOW_NORMAL)
    cv2.imwrite(output_path, original_image)
    print(f"Imagen con resultado guardada en: {output_path}")

    cv2.imshow("Resultado del Reconocimiento", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()