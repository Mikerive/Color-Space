import matplotlib.pyplot as plt
import numpy as np
import cv2

class ColorSpace:
    def __init__(self, img):
        self.img = img
    def BGR_Image_Decomposition(image):
            """
        Decomposes an image into its red, green, and blue channels.

        Args:
            image (numpy.ndarray): Input image as a numpy array.

        Returns:
            tuple: A tuple of numpy arrays representing the blue, green, and red channels of the input image.
        """
        # Split the input image into its blue, green, and red channels
        b, g, r = cv2.split(self.image)
        
        # Return the blue, green, and red channels as a tuple
        return np.array(b), np.array(g), np.array(r)