from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define the minimum colony area threshold (adjust as needed)
min_colony_area = 100

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Read the uploaded image file
    uploaded_image = request.files['image']
    
    # Check if the file has an allowed extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if uploaded_image.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension. Only JPG, JPEG, and PNG are supported'}), 400
    
    # Read the image data
    image_data = uploaded_image.read()
    
    # Convert the image data to a NumPy array
    nparr = np.fromstring(image_data, np.uint8)
    
    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a counter for the number of colonies in the current image
    colony_count = 0

    # Loop through the detected contours in the current image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_colony_area:
            colony_count += 1

    # Return the colony count as JSON response
    return jsonify({'colony_count': colony_count})

if __name__ == '__main__':
    app.run(debug=True)
