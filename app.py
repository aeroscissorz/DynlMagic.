from flask import Flask, render_template, request
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

def change_hue(img_rgb, target_hue=0):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 0] = target_hue
    img_hue_changed_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_hue_changed_rgb

def adjust_hsi_values(img_lab, hue_diff, saturation_diff, intensity_diff):
    # Adjust HSI values of the image
    img_lab[:, :, 1] += hue_diff
    img_lab[:, :, 1][img_lab[:, :, 1] > 360] -= 360  # Handle values beyond 360
    img_lab[:, :, 2] += saturation_diff
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2], 0, 100)  # Clip saturation values between 0 and 100
    img_lab[:, :, 0] += intensity_diff
    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0, 100)  # Clip intensity values between 0 and 100
    return img_lab

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def get_image_preview(image):
    img_buffer = BytesIO()
    pil_image = Image.fromarray(image)
    pil_image.save(img_buffer, format="PNG")
    img_b64 = b64encode(img_buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_b64}'

def normalize_hue(hue):
    # Normalize hue to be within [0, 360]
    hue %= 360
    return hue

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
            
            # Change the hue of the single image to (0, 0, 0)
            img_single_rgb = change_hue(img_single_rgb, target_hue=0)

            img_group = cv2.imdecode(np.frombuffer(group_image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_group = cv2.cvtColor(img_group, cv2.COLOR_BGR2RGB)

            dominant_colors_group = find_dominant_colors(img_group, num_colors=15)
            
            # Add black as a dominant color
            black_color = np.array([[0, 0, 0]], dtype=np.uint8)
            dominant_colors_group = np.vstack([dominant_colors_group, black_color])

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

    # Target only the dominant color in the single image
    target_color_lab = rgb_to_lab(np.array([find_dominant_color(img_single_rgb)]))
    dominant_color_group_lab = dominant_colors_group_lab[color_index - 1]

    # Calculate HSI difference between target color and dominant color in group image
    hue_diff = dominant_color_group_lab[1] - target_color_lab[0, 1]
    saturation_diff = dominant_color_group_lab[2] - target_color_lab[0, 2]
    intensity_diff = dominant_color_group_lab[0] - target_color_lab[0, 0]

    # Handle special case for black color
    if np.all(target_color_lab == [0, 0, 0]):
        # If the target color is black, directly adjust intensity
        img_single_lab[:, :, 0] += intensity_diff
        img_single_lab[:, :, 0] = np.clip(img_single_lab[:, :, 0], 0, 100)  # Clip intensity values between 0 and 100
    else:
        # Adjust HSI values of the single image to match the dominant color in the group image
        img_single_lab = adjust_hsi_values(img_single_lab, hue_diff, saturation_diff, intensity_diff)

    img_single_rgb_changed = lab_to_rgb(img_single_lab)

    img_buffer = BytesIO()
    pil_image = Image.fromarray(img_single_rgb_changed)
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
