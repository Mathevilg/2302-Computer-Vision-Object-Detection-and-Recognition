import cv2
from hand_detection import *

def check_image(image_path):
    # Read the image    
    image = cv2.imread(image_path)
    # Apply Gaussian blur using hand_detection.py
    blurred_image1 = gaussian_blur(image)

    # Apply Gaussian blur using cv2.gaussianBlur
    blurred_image2 = cv2.gaussianBlur(image, (5, 5), 0)

    # Compare the two blurred images
    if (blurred_image1 == blurred_image2).all():
        print("Both methods produce the same output.")
    else:
        print("The two methods produce different outputs.")