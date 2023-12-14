from flask import Flask, render_template, request
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from base64 import b64encode
from io import BytesIO

app = Flask(__name__)

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

    if request.method == 'POST':
        # Handle file uploads from the user
        single_image = request.files['single_image']
        group_image = request.files['group_image']

        if single_image and group_image and allowed_file(single_image.filename) and allowed_file(group_image.filename):
            # Read the uploaded images
            img_single = cv2.imdecode(np.frombuffer(single_image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_single_rgb = cv2.cvtColor(img_single, cv2.COLOR_BGR2RGB)

            img_group = cv2.imdecode(np.frombuffer(group_image.read(), np.uint8), cv2.IMREAD_COLOR)
            img_group = cv2.cvtColor(img_group, cv2.COLOR_BGR2RGB)

            # Get the dominant colors from img_group and convert to LAB color space
            dominant_colors_group = find_dominant_colors(img_group, num_colors=15)
            dominant_colors_group_lab = rgb_to_lab(dominant_colors_group)

            # Convert img_single to LAB color space
            img_single_lab = rgb_to_lab(img_single_rgb)

            for i in range(len(dominant_colors_group)):
                img_color_changed_lab = change_color_lab(
                    np.copy(img_single_lab),
                    rgb_to_lab(np.array([find_dominant_color(img_single_rgb)])),
                    dominant_colors_group_lab[i],
                    threshold=30
                )
                img_color_changed_rgb = lab_to_rgb(img_color_changed_lab)

                # Convert the modified image to base64 for displaying in HTML
                img_buffer = BytesIO()
                cv2.imwrite('static/images/changed_image_{}.png'.format(i), cv2.cvtColor(img_color_changed_rgb, cv2.COLOR_RGB2BGR))
                img_array = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
                img_b64 = b64encode(img_array).decode('utf-8')
                changed_images.append(img_b64)

    return render_template('index.html', changed_images=changed_images)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)
