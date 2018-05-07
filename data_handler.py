import numpy as np
import cv2
import os
import sys


def get_2d_array(im_path):
    """Return image scaled with factor 0.5 as numpy array."""
    im_color = cv2.imread(im_path)  # Read the image as a numpy array.
    im_color = im_color[32:64 + 32, 32:64 + 32]
    # Shape = (64,64,3) (x_pixels,y_pixels, color_channels)
    im_color = cv2.resize(im_color, (32,32))  # Rescale the image
    # Shape = (32,32,3) (x_pixels*scale, y_pixels*scale, color_channels)
    im = np.zeros(shape=(32,32,1))  # Create an empty array with the final shape (32,32,1)
    for i, x in enumerate(im_color):  # Fill the array
        for n, y in enumerate(x):  # Note: We cannot use cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY), because
            im[i][n][0] = (y[0] + y[1] + y[2]) / 3  # that will return an array with the shape (32,32), but we need
    return im                                       # an array with shape (32,32,1)


def get_label(name):
    """Returns label number for file name
    >>> get_label("class_10_Index_3454.jpeg")
    10
    """
    return int(name.split("_")[1])


def main():
    n_labels = 47
    n_images = 101784
    path = "./data"  # Path to images
    images = np.zeros(shape=(n_images, 32, 32, 1))  # Array with all images
    labels = np.zeros(shape=(n_images, n_labels))  # Array with all labels, one-hot encoded

    # Convert all images to numpy arrays
    for i, file in enumerate(os.listdir(path)):
        label = get_label(file)
        image = get_2d_array(os.path.join(path, file))
        images[i] = image
        labels[i, label] = 1
        print(str(i / n_images * 100) + "% done")

    # Save arrays
    np.save("nist_labels_32x32", labels)
    np.save("nist_images_32x32", images)


if __name__ == "__main__":
    main()
