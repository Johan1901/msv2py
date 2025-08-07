from flask import Flask, jsonify, request
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from src.main import process_and_store_image


app = Flask(__name__)


# MongoDB Atlas configuration
# Lee variables de .env
load_dotenv()                 
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["gatolate"]
collection = db["resultados_ejercicios"]


@app.route("/procesar", methods=["POST"])
def procesar_imagen() -> tuple:

    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    alumno = request.form.get("alumno")
    image_bytes = request.files["imagen"].read()

    result, error = process_and_store_image(image_bytes, alumno, collection)
    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
