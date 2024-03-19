import cv2
import numpy as np

def gaussian_blur(img):
    # Define the Gaussian kernel
    kernel = np.array([[1, 4, 7, 4, 1],
                        [4, 16, 26, 16, 4],
                        [7, 26, 41, 26, 7],
                        [4, 16, 26, 16, 4],
                        [1, 4, 7, 4, 1]]) / 273

    # Get the image dimensions
    height, width = img.shape[:2]
    # Create a new image to store the blurred result
    blurred_img = np.zeros_like(img)

    # Apply the Gaussian kernel to each pixel in the image
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            # Apply the kernel to the neighborhood of the pixel
            neighborhood = img[y-2:y+3, x-2:x+3]
            blurred_pixel = np.sum(neighborhood * kernel)
            # Set the blurred pixel value in the new image
            blurred_img[y, x] = blurred_pixel
    return blurred_img


def preprocess(img) :
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

# def extract_features(img) :
    # Compute the Histogram of Oriented Gradients (HOG) descriptor for each region of interest (ROI) in the image. Define appropriate ROIs where hands are likely to be present.
    # Return the HOG descriptors as a feature vector
