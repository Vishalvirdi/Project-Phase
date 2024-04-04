from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

min_colony_area = 100

@app.route('/', methods=['POST','GET'])
def analyze_image():
    # Check if image data is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Read the image data from the request
    image_file = request.files['image']

    # Convert image data to OpenCV format
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Failed to read image'}), 400

    colony_count = process_image(image)

    # Process the image
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for PIL
    pil_image = Image.fromarray(processed_image)

    # Convert PIL image to base64 string
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({'colony_count': colony_count, 'processed_image': encoded_image})

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Employ Canny edge detection and contour finding
    edges = cv2.Canny(blurred, threshold1=30, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_colony_area:
            colony_count += 1
            # Assuming processing involves drawing on the image (modify as needed)
            cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)  # Draw red contour

    return colony_count

if __name__ == '__main__':
    app.run(debug=True)
