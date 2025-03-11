import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# Load images (using URLs - you'll need to download them first)
healthy_image_url = 'https://suzannerbanks.blog/wp-content/uploads/2013/01/tomato-leaf.jpg'
diseased_image_url = 'https://yardandgarden.extension.iastate.edu/files/inline-images/26.jpg'

# Download the images (you'll need to implement this part)
# ... (Code to download images from URLs and save them locally)

# Load the downloaded images (replace with the local paths where you saved them)
healthy_image = cv2.imread("tomato-leaf.jpg", cv2.IMREAD_GRAYSCALE)  # Replace with the actual local path
diseased_image = cv2.imread("tomato.jpeg", cv2.IMREAD_GRAYSCALE)  # Replace with the actual local path

# Calculate GLCM
distances = [1]  # Distance between pixels
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles to consider
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

glcm_healthy = graycomatrix(healthy_image, distances, angles, levels=256, symmetric=True, normed=True)
glcm_diseased = graycomatrix(diseased_image, distances, angles, levels=256, symmetric=True, normed=True)

# Calculate GLCM properties
for prop in properties:
    feature_healthy = graycoprops(glcm_healthy, prop)
    feature_diseased = graycoprops(glcm_diseased, prop)
    print(f'{prop}: Healthy = {feature_healthy}, Diseased = {feature_diseased}')

# Visualize GLCM (example for contrast)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(glcm_healthy[:, :, 0, 0], cmap='gray')
plt.title('Healthy Leaf GLCM (Contrast)')
plt.subplot(1, 2, 2)
plt.imshow(glcm_diseased[:, :, 0, 0], cmap='gray')
plt.title('Diseased Leaf GLCM (Contrast)')
plt.show()