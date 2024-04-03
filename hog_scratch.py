import cv2
import numpy as np

def compute_gradients(gray):
    # Convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    
    # Compute gradient magnitude and angle
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    
    return magnitude, angle

def compute_histograms(magnitude, angle, cells_per_block=(2, 2), bins=9):
    cell_size = 8
    hist = []

    # Calculate the number of cells in x and y directions
    cell_rows, cell_cols = magnitude.shape[0] // cell_size, magnitude.shape[1] // cell_size

    for r in range(cell_rows):
        for c in range(cell_cols):
            cell_magnitude = magnitude[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
            cell_angle = angle[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]

            # Compute histogram for each cell
            hist_block = np.zeros(bins)
            for i in range(cell_size):
                for j in range(cell_size):
                    angle_val = cell_angle[i, j]
                    bin_idx = int(angle_val / (180 / bins))

                    # Weighted voting
                    vote_frac, bin_frac = np.modf(angle_val / (180 / bins))
                    vote_frac = 1 - np.abs(vote_frac - 0.5)  # Triangular weighting function
                    hist_block[(bin_idx + int(bin_frac)) % bins] += vote_frac * cell_magnitude[i, j]
                    hist_block[(bin_idx + int(bin_frac) + 1) % bins] += (1 - vote_frac) * cell_magnitude[i, j]

            hist.append(hist_block)

    return np.array(hist).flatten()

def descriptor_scratch(image):
    # Compute gradients
    magnitude, angle = compute_gradients(image)
    
    # Compute HOG descriptors
    hog_features = compute_histograms(magnitude, angle)
    
    return hog_features
