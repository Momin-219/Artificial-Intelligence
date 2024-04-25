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
        image_path2 = data.get('image_path2')
        print( image_path1)
        print( image_path2)
        result_similarity = DeepFace.find(image_path1,image_path2)
        result_expression = DeepFace.analyze(image_path1 ,actions=['gender','age','race','emotion'])
        result_similarity_dicts = [df.to_dict(orient='records') for df in result_similarity]
        result = {'faces_detected': image_path1, 'similarity': result_similarity_dicts}
        result['expression']=[result_expression]
        print(result)
        '''
        result = {
          'similarity': result_similarity_dicts
        }
        '''
        return result

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
