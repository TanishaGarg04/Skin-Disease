from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response, flash, session
import os
import io
import numpy as np
from PIL import Image
import logging
import traceback
import cv2
import glob
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Delete existing database and create new one
with app.app_context():
    db.drop_all()  # This will delete all existing tables
    db.create_all()  # This will create new tables with updated schema
    
    # Create admin user
    admin = User(username='admin', email='admin@skinarmour.com', is_admin=True)
    admin.set_password('admin123')  # Change this password
    db.session.add(admin)
    db.session.commit()

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
    img_array = np.array(uploaded_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    hist1 = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    min_dist = float('inf')
    best_match = None
    image_paths = glob.glob(os.path.join(dataset_folder, '*', '*.*'))
    
    # Process images in batches
    batch_size = 10
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        for img_path in batch_paths:
            try:
                dataset_img = cv2.imread(img_path)
                if dataset_img is None:
                    continue
                    
                # Resize image
                dataset_img = cv2.resize(dataset_img, (224, 224))
                
                # Calculate histogram
                hist2 = cv2.calcHist([dataset_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.normalize(hist2, hist2).flatten()
                
                # Compare histograms
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
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email and password:
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                session['user_id'] = user.id
                session['is_admin'] = user.is_admin  # Store admin status in session
                flash('Successfully signed in!', 'success')
                
                # Check if user is admin
                if user.is_admin and user.email == 'admin@skinarmour.com':
                    return redirect(url_for('admin'))
                else:
                    return redirect(url_for('detect'))
            else:
                flash('Invalid email or password!', 'error')
        else:
            flash('Please fill in all fields!', 'error')
        return redirect(url_for('signin'))
            
    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('Successfully logged out!', 'success')
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if username and email and password and password == confirm_password:
            # Check if user already exists
            if User.query.filter_by(email=email).first():
                flash('Email already registered!', 'error')
                return redirect(url_for('signup'))
            if User.query.filter_by(username=username).first():
                flash('Username already taken!', 'error')
                return redirect(url_for('signup'))
            
            # Create new user
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            
            try:
                db.session.add(new_user)
                db.session.commit()
                flash('Successfully signed up! Please sign in.', 'success')
                return redirect(url_for('signin'))
            except Exception as e:
                db.session.rollback()
                flash('Registration failed! Please try again.', 'error')
        else:
            flash('Registration failed! Please check your details.', 'error')
        return redirect(url_for('signup'))
            
    return render_template('signup.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.pop('user_id', None)
        session.pop('is_admin', None)
        flash('Please sign in to continue.', 'error')
        return redirect(url_for('signin'))

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
            imagePil = Image.open(io.BytesIO(file.read()))
            best_match, similarity = find_most_similar_image(imagePil, DATASET_LINK)
            if best_match:
                disease_name = os.path.basename(os.path.dirname(best_match))
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

@app.route('/admin')
def admin():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('admin_login'))
    
    user = User.query.get(session['user_id'])
    # Strictly check for admin email
    if not user or not user.is_admin or user.email != 'admin@skinarmour.com':
        session.pop('user_id', None)
        session.pop('is_admin', None)
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin_login'))
    
    # Get all users from database
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email and password:
            # Strictly check for admin email
            if email != 'admin@skinarmour.com':
                flash('Invalid admin credentials!', 'error')
                return redirect(url_for('admin_login'))
                
            user = User.query.filter_by(email=email).first()
            if user and user.is_admin and user.check_password(password):
                session['user_id'] = user.id
                session['is_admin'] = user.is_admin
                flash('Successfully logged in as admin!', 'success')
                return redirect(url_for('admin'))
            else:
                flash('Invalid admin credentials!', 'error')
        else:
            flash('Please fill in all fields!', 'error')
        return redirect(url_for('admin_login'))
            
    return render_template('admin_login.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
