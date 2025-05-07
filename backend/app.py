# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
import cv2
import numpy as np
import pickle

app = Flask(__name__)
CORS(app, origins=["https://facerecognitionv.vercel.app"])

# Cargar rutas de base de datos (las imágenes de referencia)
with open('deepface_database_paths.pkl', 'rb') as f:
    database_paths = pickle.load(f)

# Puedes cargar las imágenes una vez en memoria si lo prefieres
print(f"Base de datos cargada con {len(database_paths)} personas.")

@app.route('/')
def index():
    return '✅ DeepFace Backend Running!'

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.find(img_path=img, db_path='database_images', enforce_detection=False)
        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            name = os.path.basename(os.path.dirname(identity))
            return jsonify({'identity': name})
        else:
            return jsonify({'identity': 'unknown'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Recognition failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
