import cv2
import numpy as np
import matplotlib as plt
from skimage.transform import resize

__all__ = ['Filters']

class Filters:
    def __init__(self, img = None, img_name = 'default'):
        self.img = np.asarray(img)
        self.img_name = img_name
        
    def reduce_size(self, factor=2):
        img = np.array(self.img).astype(np.int)
        
        # Resize the image to half its original size
        height, width = img.shape[:factor]
        img_resized = resize(img, (height//factor, width //
                            factor), anti_aliasing=False)
        
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
        return img_resized

    def contrast_stretch(self):
        # Convert the image to a numpy array
        img = np.array(self.img).astype(np.int)

        # Calculate the minimum and maximum values of the image
        min_val = np.min(img)
        max_val = np.max(img)

        # New minimum and maximum values
        output_min = 0
        output_max = 255

        # Contrast Stretch = Current Delta * Stretching factor + new_min
        img = (img - min_val) * ((output_max - output_min) /
                                (max_val - min_val)) + output_min

        # Convert the image back to 8-bit unsigned integer format
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        self.img = img
        return img

    def gamma_correction(self, gamma=1.0, alpha=0.0):
        # Convert the image to a numpy array
        img = np.array(self.img).astype(np.int)

        # Normalize the image to the range [0, 1]
        img = img / 255.0

        # Apply the gamma correction
        img = np.power(img+alpha, gamma)

        # Scale the image back to the range [0, 255]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        self.img = img
        return img

    def histogram_equalization(self):
        
        # Convert the image to a numpy array
        img = np.array(self.img).astype(np.int)
        
        # Calculate the cdf
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()

        # Calculate the normalized cdf
        cdf_normalized = cdf / cdf.max()

        # Map the pixel values to the normalized cumulative distribution
        img = np.interp(self.img, bins[:-1], cdf_normalized)

        # Convert the image back to 8-bit unsigned integer format
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        
        self.img = img
        return img


