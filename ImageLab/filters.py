import numpy as np
from skimage.transform import resize

from .noisegenerators import NoiseOverlay
from .imageutils import *
from .ImageProcessing import Convolution

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
        img = np.array(self.img).astype(np.int8)
        
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

    def sharpening(self, kernel=np.array([[0, -1, 0],
                                          [-1, 5, -1],
                                          [0, -1, 0]])):

        # Normalize the kernel
        kernel = kernel / np.sum(kernel)

        # Pad the image with zeros to handle border pixels
        padded_image = np.pad(self.img, 1, 'constant', constant_values=0)

        # Apply the kernel to the image using convolution
        sharp = np.zeros_like(self.img)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                sharp[i, j] = np.sum(kernel * padded_image[i:i+3, j:j+3])

        # Clip the output image to ensure that pixel values are within [0, 255]
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        
        ImagePlotter(sharp).plot_image_with_histogram(
            title=f'{self.img_name}', cmap='Greys')
        
        # img = self.img - sharp
        
        # ImagePlotter(img).plot_image_with_histogram(
        #     title=f'{self.img_name}', cmap='Greys')
        
        # img = self.img + img
        
        # ImagePlotter(img).plot_image_with_histogram(
        #     title=f'{self.img_name}', cmap='Greys')
    
    def unsharp_masking(self, mean, var):
        noise = NoiseOverlay(self.img).add_gaussian_noise(mean, var)
        img = np.copy(self.img)
        sharp = img - noise
        ImagePlotter(sharp).plot_image_with_histogram(
            title=f'{self.img_name} {mean}:{var}', cmap='Greys')
        sharpened = img + sharp
        ImagePlotter(sharpened).plot_image_with_histogram(
            title=f'{self.img_name} {mean}:{var}', cmap='Greys')
        return sharpened
    