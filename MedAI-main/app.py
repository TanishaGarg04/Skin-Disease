from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import os
import io
import numpy as np
from PIL import Image
import logging
import traceback
import cv2
import glob

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Dataset link
DATASET_LINK = r"C:\Users\TANISHA\OneDrive\Desktop\dataset\skin disease\skin-disease-datasaet\test_set"

def analyze_image(image):
    """Simple image analysis based on color and texture"""
    try:
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_cv = cv2.resize(img_cv, (224, 224))
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Calculate average color
        avg_color = np.mean(hsv, axis=(0,1))
        
        # Calculate color standard deviation
        std_color = np.std(hsv, axis=(0,1))
        
        # Calculate edge density
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Simple analysis based on color and edge features
        if edge_density > 0.1:
            if avg_color[0] > 100:  # Hue
                return 0, 85.0  # Class 0 with 85% confidence
            else:
                return 1, 75.0  # Class 1 with 75% confidence
        else:
            if avg_color[1] > 100:  # Saturation
                return 2, 80.0  # Class 2 with 80% confidence
            else:
                return 3, 70.0  # Class 3 with 70% confidence
                
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}\n{traceback.format_exc()}")
        return 0, 0.0

def find_most_similar_image(uploaded_image, dataset_folder):
    # Convert uploaded image to OpenCV format and resize
    img_array = np.array(uploaded_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    # Calculate histogram for uploaded image
    hist1 = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    min_dist = float('inf')
    best_match = None

    # Loop through dataset images
    for img_path in glob.glob(os.path.join(dataset_folder, '*', '*.*')):
        try:
            dataset_img = cv2.imread(img_path)
            if dataset_img is None:
                continue
            dataset_img = cv2.resize(dataset_img, (224, 224))
            hist2 = cv2.calcHist([dataset_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()
            dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            if dist < min_dist:
                min_dist = dist
                best_match = img_path
        except Exception as e:
            continue

    return best_match, min_dist

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        return redirect(url_for('detect'))
    return render_template('signin.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html', dataset_link=DATASET_LINK)
    
    try:
        if 'file' not in request.files:
            return make_response(jsonify({
                'error': True,
                'message': 'No file uploaded',
                'dataset_link': DATASET_LINK
            }), 400)
        
        file = request.files['file']
        if file.filename == '':
            return make_response(jsonify({
                'error': True,
                'message': 'No file selected',
                'dataset_link': DATASET_LINK
            }), 400)
        
        try:
            # Process the image
            imagePil = Image.open(io.BytesIO(file.read()))
            
            # Find most similar image in dataset
            best_match, similarity = find_most_similar_image(imagePil, DATASET_LINK)
            
            # Extract disease name from the best match path
            if best_match:
                # Get the folder name which contains the disease name
                disease_name = os.path.basename(os.path.dirname(best_match))
                # Calculate accuracy (convert similarity to percentage)
                accuracy = (1 - similarity) * 100
                accuracy = round(accuracy, 2)
            else:
                disease_name = "Unknown"
                accuracy = 0
            
            return make_response(jsonify({
                'error': False,
                'detected': True,
                'disease': disease_name,
                'accuracy': accuracy,
                'img_path': file.filename
            }), 200)
                
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}\n{traceback.format_exc()}")
            return make_response(jsonify({
                'error': True,
                'message': 'Error processing image',
                'dataset_link': DATASET_LINK
            }), 500)
            
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({
            'error': True,
            'message': 'Server error occurred',
            'dataset_link': DATASET_LINK
        }), 500)

@app.route('/clinics')
def clinics():
    return render_template('clinics.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
