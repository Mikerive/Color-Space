import matplotlib.pyplot as plt
import numpy as np
import cv2
from imageplotter import ImagePlotter
from noisegenerators import gaussian_noise

class ColorSpace:
    def __init__(self, img):
        self.img = img
    def BGR_Image_Decomposition(self):
        # Split the input image into its blue, green, and red channels
        b, g, r = cv2.split(self.image)
        
        # Return the blue, green, and red channels as a tuple
        return np.array(b), np.array(g), np.array(r)
    
    def show_grayscale(self, ratio=[0.2989, 0.5870, 0.1140]):
        # Load the image and convert it to grayscale
        image = plt.imread(self.img)
        gray_image = np.dot(image[..., :3], ratio)

        # Print the grayscale pixel values
        plt.imshow(gray_image, cmap='gray')
    

        # # Split the RGB image into its red, green, and blue channels
        # r, g, b = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]

        # # Merge the noisy channels back into an RGB image
        # noisy_image = np.stack([r, g, b], axis=2)
    
    

        
    
