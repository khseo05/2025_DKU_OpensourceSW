# Python code for Flask web application
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

def apply_smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed_image

def apply_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return sepia_image

def apply_sharpening(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def apply_embossing(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return embossed_image

def apply_invert(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def apply_posterize(image):
    posterized_image = cv2.convertScaleAbs(image, alpha=(1/32) * 32, beta=0)
    return posterized_image


def apply_thresholding(image):
    ret, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresholded_image

def apply_filter(image, filter_name):
    if filter_name == 'smoothing':
        return apply_smoothing(image)
    elif filter_name == 'edge_detection':
        return apply_edge_detection(image)
    elif filter_name == 'grayscale':
        return apply_grayscale(image)
    elif filter_name == 'sepia':
        return apply_sepia(image)
    elif filter_name == 'sharpening':
        return apply_sharpening(image)
    elif filter_name == 'embossing':
        return apply_embossing(image)
    elif filter_name == 'invert':
        return apply_invert(image)
    elif filter_name == 'posterize':
        return apply_posterize(image)
    elif filter_name == 'thresholding':
        return apply_thresholding(image)
    else:
        return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/filter', methods=['POST'])
def filter():
    image = request.files['image']
    npimg = np.fromstring(image.read(), np.uint8)
    cvimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    filter_name = request.form['filter']
    filtered_image = apply_filter(cvimg, filter_name)
    _, buffer = cv2.imencode('.jpg', filtered_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return 'data:image/jpeg;base64,' + encoded_image

if __name__ == '__main__':
    app.run(debug=True)
