from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json
import os


app = Flask(__name__)

PRESET_PATH = './presets' # 저장할 폴더 지정

if not os.path.exists(PRESET_PATH):
    os.makedirs(PRESET_PATH)

# Image filters
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

# 250511 정다훈 추가
def apply_bilateral_filter(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    return bilateral_filtered_image


# 250511 정다훈 추가
def apply_laplacian(image):
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian_image


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
    elif filter_name == 'bilateral_filter':
        return apply_bilateral_filter(image)
    elif filter_name == 'laplacian':
        return apply_laplacian(image)
    else:
        return image


# Noise cancel    
def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
   
def apply_noise(image, filter_name):
    if filter_name == 'median_filter':
        return apply_median_filter(image)
    elif filter_name == 'gaussian_filter':
        return apply_gaussian_filter(image)
    else:
        return image


# image adjustment
def apply_adjustments(image, adjustment_type, adjustment_value):

    if adjustment_type == 'brightness':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] + adjustment_value, 0, 255)
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif adjustment_type == 'saturation':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] + adjustment_value, 0, 255)
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif adjustment_type == 'contrast':
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
        adjusted_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    else:
        adjusted_image = image

    return adjusted_image


# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-adjustments')
def chane_img():
    return render_template('image-adjustments.html')

@app.route('/remove-noise')
def remove_noise():
    return render_template('remove-noise.html')

@app.route('/image-adjustments', methods=['POST'])
def image_adjustments():
    image_file = request.files['image']
    image_array = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    adjustment_type = request.form['adjustmentType']
    adjustment_value = int(request.form['adjustmentValue'])

    print("Adjustment type: " + adjustment_type + " Value: " + str(adjustment_value))

    # Apply adjustments
    adjusted_image = apply_adjustments(image, adjustment_type, adjustment_value)

    # Encode
    _, image_bytes = cv2.imencode('.jpg', adjusted_image)
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    return 'data:image/jpeg;base64,' + encoded_image 

#250525 강현서 수정 - 필터 경고문
conflict_filters = {"grayscale", "invert", "posterize", "thresholding", "edge_detection"}
def check_conflicting_filter(selected_filters):
    conflicts = []
    for f in selected_filters:
        if f in conflict_filters and len(selected_filters) > 1:
            conflicts.append(f)

    return conflicts

# 250511 강현서 수정

@app.route('/filter', methods=['POST'])
def filter():
    print("filter open")
    image = request.files['image']
    npimg = np.frombuffer(image.read(), np.uint8)
    cvimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    filter_names = json.loads(request.form['filter'])
    print(f"필터 목록: {filter_names}") 

    # 충돌 필터 체크
    conflicts = check_conflicting_filter(filter_names)

    for filter_name in filter_names: 
        cvimg = apply_filter(cvimg, filter_name)
        print(f"필터 적용:", {filter_name})

    _, buffer = cv2.imencode('.jpg', cvimg)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    response = {'dataUrl': f'data:image/jpeg;base64,{encoded_image}',
                'conflicts': list(conflicts)}
    return jsonify(response)

@app.route('/remove-noise', methods=['POST'])
def noise():
    print("Noise open")
    noise_image = request.files['image']
    npimg = np.fromstring(noise_image.read(), np.uint8)
    cvimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    filter_name = request.form['filter']
    filtered_image = apply_noise(cvimg, filter_name)
    _, img = cv2.imencode('.jpg', filtered_image)
    noise_image = base64.b64encode(img).decode('utf-8')
    return 'data:image/jpeg;base64,' + noise_image

# 250520 강현서 수정
@app.route('/save-preset', methods=['POST'])
def save_preset():
    data = request.json
    name = data.get('name')  # preset 이름
    filters = data.get('filters')  # 리스트 형태의 필터 목록

    if not name or not filters:
        return {'status': 'error', 'message': 'Invalid preset'}, 400

    with open(os.path.join(PRESET_PATH, f'{name}.json'), 'w') as f:
        json.dump(filters, f)

    return {'status': 'success', 'message': f'Preset {name} saved.'}

# 250520 강현서 수정 - 나만의 필터 로직 
@app.route('/list-presets', methods=['GET'])
def list_presets():
    preset_dir = 'presets'  
    if not os.path.exists(preset_dir):
        return jsonify([])

    preset_files = [f[:-5] for f in os.listdir(preset_dir) if f.endswith('.json')]
    return jsonify(preset_files)

@app.route('/load-preset', methods=['GET'])
def load_preset():
    preset_name = request.args.get('name')

    if not preset_name:
        return jsonify({'오류': '프리셋 이름 포함 X'}), 400
    try:
        with open(f'presets/{preset_name}.json', 'r') as f:
            preset_data = json.load(f)
        return jsonify(preset_data)
    except FileNotFoundError:
        return jsonify({'오류': '프리샛 발견 X'}), 404
    
if __name__ == '__main__':
    app.run(debug=True)