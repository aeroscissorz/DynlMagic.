from flask import Flask, render_template, request, send_file
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

def change_dominant_color_lab(img_lab, target_color_lab, replacement_color_lab, threshold=15, intensity_threshold_factor=50, blending_ratio=0.5):
    distance = np.linalg.norm(img_lab[:, :, 1:] - target_color_lab[:, 1:], axis=-1)
    
    # Adjust intensity threshold based on the average intensity of the target color
    target_intensity = target_color_lab[0, 0]
    intensity_threshold = target_intensity / intensity_threshold_factor
    
    # Consider all pixels for replacement (no intensity threshold)
    mask = distance < threshold
    
    # Increase blending power for all pixels (adjust the ratio as needed)
    img_lab[mask, 1:] = blending_ratio * replacement_color_lab[1:] + (1 - blending_ratio) * img_lab[mask, 1:]
    
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

    img_color_changed_lab = change_dominant_color_lab(
        np.copy(img_single_lab),
        rgb_to_lab(np.array([find_dominant_color(img_single_rgb)])),
        dominant_colors_group_lab[color_index - 1],
        threshold=35,
        intensity_threshold_factor=50,
        blending_ratio=0.7
    )
    img_color_changed_rgb = lab_to_rgb(img_color_changed_lab)

    img_buffer = BytesIO()
    pil_image = Image.fromarray(img_color_changed_rgb)
    pil_image.save(img_buffer, format="PNG")
    img_b64 = b64encode(img_buffer.getvalue()).decode('utf-8')

    changed_image = f'data:image/png;base64,{img_b64}'
    color_changed = True

    # Save the image temporarily for downloading
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'changed_image.png')
    pil_image.save(temp_file_path)

    return render_template('index.html', changed_image=changed_image, single_image_preview=get_image_preview(img_single_rgb),
                           group_image_preview=get_image_preview(img_group),
                           dominant_colors_group=[list(color) for color in dominant_colors_group_lab],
                           color_changed=color_changed, temp_file_path=temp_file_path)

@app.route('/download_changed_image')
def download_changed_image():
    # Provide the option to download the changed image
    temp_file_path = request.args.get('temp_file_path', default='', type=str)

    if os.path.exists(temp_file_path):
        return send_file(temp_file_path, as_attachment=True)
    else:
        return "Error: The file does not exist."

if __name__ == '__main__':
    app.run(debug=True)
