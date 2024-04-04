from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

min_colony_area = 100

    
@app.route('/hello', methods=['GET'])
def get():
    res.send("hello")


@app.route('/', methods=['POST'])
def analyze_image():

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    uploaded_image = request.files['image']
    

    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if uploaded_image.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension. Only JPG, JPEG, and PNG are supported'}), 400
    
    image_data = uploaded_image.read()
    
    nparr = np.fromstring(image_data, np.uint8)
    
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_colony_area:
            colony_count += 1
    
    return jsonify({'colony_count': colony_count})
    
    
    

    

if __name__ == '__main__':
    app.run(debug=True)
