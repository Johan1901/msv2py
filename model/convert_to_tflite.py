import tensorflow as tf

# Ruta del modelo entrenado (.h5)
MODEL_PATH = 'model/math_symbol_recognizer.h5'

# Ruta de salida del modelo .tflite
TFLITE_MODEL_PATH = 'model/math_symbol_recognizer.tflite'

# Cargar el modelo .h5
model = tf.keras.models.load_model(MODEL_PATH)

# Convertir a TensorFlow Lite
print("\n🔁 Convirtiendo el modelo a TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar como .tflite
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Modelo convertido y guardado como: {TFLITE_MODEL_PATH}")
