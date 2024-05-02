from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/filter', methods=['POST'])
def filter():
    # Get uploaded image
    image = request.files['image']
    npimg = np.fromstring(image.read(), np.uint8)
    cvimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Apply filter 
    gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)

    # Encode filtered image to base64
    _, buffer = cv2.imencode('.jpg', gray)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return 'data:image/jpeg;base64,' + encoded_image

if __name__ == '__main__':
    app.run(debug=True)
