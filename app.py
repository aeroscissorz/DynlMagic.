from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from base64 import b64encode
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_IMAGES_FOLDER = 'static-images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

img_single_lab = None
img_single_rgb = None
dominant_colors_group_lab = None
img_group = None
color_changed = False

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

def change_color_lab(img_lab, target_color_lab, replacement_color_lab, threshold=60, alpha=2):
    """
    Change color of the image in LAB color space with a smooth transition.
    
    :param img_lab: Original image in LAB color space.
    :param target_color_lab: Target color in LAB space to be replaced.
    :param replacement_color_lab: Replacement color in LAB space.
    :param threshold: Distance threshold for color replacement.
    :param alpha: Controls the intensity of the color change (0.0 to 1.0).
    :return: Image with color changed in LAB color space.
    """
    distance = np.linalg.norm(img_lab - target_color_lab, axis=-1)
    mask = distance < threshold
    img_lab[mask] = alpha * replacement_color_lab + (1 - alpha) * img_lab[mask]
    return img_lab

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def get_image_preview(image):
    img_buffer = BytesIO()
    pil_image = Image.fromarray(image)
    pil_image.save(img_buffer, format="PNG")
    img_b64 = b64encode(img_buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_b64}'

@app.route('/', methods=['GET', 'POST'])
def index():
    global img_single_rgb, img_single_lab, dominant_colors_group_lab, img_group, color_changed

    single_image_preview = None
    group_image_preview = None
    dominant_colors_group = []

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

            single_image_preview = get_image_preview(img_single_rgb)
            group_image_preview = get_image_preview(img_group)

            img_single_lab = rgb_to_lab(img_single_rgb)
            color_changed = False

    return render_template('index.html', single_image_preview=single_image_preview,
                           group_image_preview=group_image_preview, dominant_colors_group=dominant_colors_group,
                           changed_image=None, color_changed=color_changed)

@app.route('/show_changed_image', methods=['POST'])
def show_changed_image():
    global img_single_lab, img_single_rgb, dominant_colors_group_lab, img_group, color_changed

    color_index = int(request.form['color_index'])

    img_color_changed_lab = change_color_lab(
        np.copy(img_single_lab),
        rgb_to_lab(np.array([find_dominant_color(img_single_rgb)])),
        dominant_colors_group_lab[color_index - 1],
        threshold=30,
        alpha=0.5
    )
    img_color_changed_rgb = lab_to_rgb(img_color_changed_lab)

    img_buffer = BytesIO()
    pil_image = Image.fromarray(img_color_changed_rgb)
    pil_image.save(img_buffer, format="PNG")
    img_b64 = b64encode(img_buffer.getvalue()).decode('utf-8')

    changed_image = f'data:image/png;base64,{img_b64}'
    color_changed = True

    return render_template('index.html', changed_image=changed_image, single_image_preview=get_image_preview(img_single_rgb),
                           group_image_preview=get_image_preview(img_group),
                           dominant_colors_group=[list(color) for color in dominant_colors_group_lab],
                           color_changed=color_changed)

if __name__ == '__main__':
    app.run(debug=True)
