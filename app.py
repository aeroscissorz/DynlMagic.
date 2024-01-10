from flask import Flask, render_template, request
import cv2
import numpy as np
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
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    return img_lab

def lab_to_rgb(img_lab):
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
    return img_rgb

def color_transfer(source, target):
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2Lab)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab)

    # Compute mean and standard deviation of each channel
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    # Perform color transfer
    result_lab = (target_std / source_std) * (source_lab - source_mean) + target_mean
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_Lab2RGB)
    return result_rgb

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

    # Target only the dominant color in the single image
    target_color_lab = rgb_to_lab(np.array([find_dominant_color(img_single_rgb)]))
    img_color_changed_rgb = color_transfer(np.copy(img_single_rgb), dominant_colors_group_lab[color_index - 1])

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
