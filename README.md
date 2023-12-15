# DynlMagic

![UI Screenshot](https://github.com/aeroscissorz/DynlMagic.github.io/blob/main/Screenshot%202023-12-14%20213827.png)

## How to Run

To run the Image Processing Web App, follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/aeroscissorz/DynlMagic.github.io.git
   ```

2. Navigate to the project directory:

   ```bash
   cd DynlMagic.github.io
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

## What it Does

The Image Processing Web App is designed to perform color transformation on a single image based on the dominant colors extracted from a group image. 

## Use Case

The web app can be useful in scenarios where you want to harmonize the color scheme of a single image with the dominant colors of a group of images. For example, it can be used in graphic design or image editing tasks.

## Limitations

- The application currently supports only images with three color channels (RGB).
- The color transformation is based on a simple algorithm and may not be suitable for all use cases.

## Known Issues

- No known issues at the moment.

## Tech Stacks with Version
Flask: version 2.1.0
OpenCV: version 4.5.3
NumPy: version 1.21.1
scikit-image: version 0.18.3
Matplotlib: version 3.4.3

