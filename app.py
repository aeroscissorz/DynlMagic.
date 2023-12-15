from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from base64 import b64encode
from io import BytesIO
from PIL import Image

app = Flask(__name__)

img_single_lab = None
img_single_rgb = None
dominant_colors_group_lab = None  # Initialize as a global variable

def find_dominant_colors(img, num_colors=15):
    pixels = img.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(np.float32(pixels), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_colors = np.uint8(centers)
    return dominant_colors

def find_dominant_color(img):
    pixels = img.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(np.float32(pixels), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = np.uint8(centers[0])
    return dominant_color

def rgb_to_lab(img_rgb):
    img_lab = rgb2lab(img_rgb)
    return img_lab

def lab_to_rgb(img_lab):
    img_rgb = lab2rgb(img_lab)
    img_rgb = (img_rgb * 255).astype(np.uint8)
    return img_rgb

def change_color_lab(img_lab, target_color_lab, replacement_color_lab, threshold=60):
    distance = np.linalg.norm(img_lab - target_color_lab, axis=-1)
    mask = distance < threshold
    img_lab[mask] = replacement_color_lab
    return img_lab

@app.route('/', methods=['GET', 'POST'])
def index():
    changed_images = []
    dominant_colors = []

    global img_single_lab, img_single_rgb, dominant_colors_group_lab

    if request.method == 'POST':
        single_image = request.files['single_image']
        group_image = request.files['group_image']

        if single_image and group_image and allowed_file(single_image.filename) and allowed_file(group_image.filename):
            img_single = cv2.imdecode(np.frombuffer(single_image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_single_rgb = cv2.cvtColor(img_single, cv2.COLOR_BGR2RGB)

            img_group = cv2.imdecode(np.frombuffer(group_image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_group = cv2.cvtColor(img_group, cv2.COLOR_BGR2RGB)

            dominant_colors_group = find_dominant_colors(img_group, num_colors=15)
            dominant_colors_group_lab = rgb_to_lab(dominant_colors_group)

            print("Dominant Colors:")
            for color in dominant_colors_group:
                print(color)
                dominant_colors.append(color.tolist())

            img_single_lab = rgb_to_lab(img_single_rgb)

    return render_template('index.html', dominant_colors=dominant_colors)

@app.route('/show_changed_image', methods=['POST'])
def show_changed_image():
    global img_single_lab, img_single_rgb, dominant_colors_group_lab

    color_index = int(request.form['color_index'])

    img_color_changed_lab = change_color_lab(
        np.copy(img_single_lab),
        rgb_to_lab(np.array([find_dominant_color(img_single_rgb)])),
        dominant_colors_group_lab[color_index],
        threshold=30
    )
    img_color_changed_rgb = lab_to_rgb(img_color_changed_lab)

    img_buffer = BytesIO()
    pil_image = Image.fromarray(img_color_changed_rgb)
    pil_image.save(img_buffer, format="PNG")
    img_b64 = b64encode(img_buffer.getvalue()).decode('utf-8')

    return f'<img src="data:image/png;base64,{img_b64}" alt="Changed Image">'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)
