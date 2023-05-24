# this program will display a GUI for the user to manually calibrate the camera

import cv2
import numpy as np
import imutils

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

def main():
    # image path to be used for calibration
    image_path = '/Users/cta/Downloads/windows.jpg'

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # check if image is loaded correctly
    if image is None:
        print('Could not open or find the image')
        exit(0)

    # Convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the distortion coefficients and the camera matrix
    dist_coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    camera_matrix = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    im = ax.imshow(image)

    # Create the sliders
    slider_axes = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider1 = Slider(slider_axes, 'k1', -0.2, 0.2, valinit=0)
    slider2 = Slider(slider_axes, 'k2', -0.2, 0.4, valinit=0)
    slider3 = Slider(slider_axes, 'p1', -0.2, 0.6, valinit=0)
    slider4 = Slider(slider_axes, 'p2', -0.2, 0.8, valinit=0)
    slider5 = Slider(slider_axes, 'k3', -0.2, 1.2, valinit=0)

    # Define a function that will be called when the sliders are moved
    def update(val):
        # Get the current values of the sliders
        k1 = slider1.val
        k2 = slider2.val
        p1 = slider3.val
        p2 = slider4.val
        k3 = slider5.val

        # Update the distortion coefficients
        dist_coeffs[0] = k1
        dist_coeffs[1] = k2
        dist_coeffs[2] = p1
        dist_coeffs[3] = p2
        dist_coeffs[4] = k3

        # Compute the undistorted image
        h, w = image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Update the image
        im.set_data(undistorted_image)
        fig.canvas.draw_idle()

    # Attach the update function to the


    # Show the figure
    plt.show()

if __name__ == '__main__':
    main()