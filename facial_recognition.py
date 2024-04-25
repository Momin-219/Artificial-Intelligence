from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

@app.route('/process_images', methods=['POST'])
def process_images():

        data = request.json
        #print("Received data:", data)
        image_path1 = data.get('image_path1')
        print( image_path1)
        result_expression = DeepFace.analyze(image_path1 ,actions=['gender','age','race','emotion'])
        print( result_expression)
        result = {
            'expression': result_expression,
       #    'similarity': result_similarity
        }
        return jsonify(result)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
