import numpy as np
from skimage.transform import resize
from PIL import Image

from .noisegenerators import NoiseOverlay
from .imageutils import *
from .ImageProcessing import Convolution

__all__ = ['Filters']

class Filters:
    def __init__(self, image_path, folder_name, img_name, hist = False):
        self.image_path = image_path
        self.folder_name = folder_name
        self.img_name = img_name
        self.hist = hist
    
    def process(self, operator):
        # Open the input image
        image = Image.open(self.image_path)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
        
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array)
        
        output = np.clip(output, 0, 255).astype(np.uint8)

        path = ImageUtil(output).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{self.class_name}.png")
        
        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(f'{self.img_name}_{class_name}')
            
        else:
            ImagePlotter(output).plot_image(f'{self.img_name}_{class_name}')
        
        return output, path
        
    def reduce_size(self, factor=2):
        img = np.array(self.img).astype(np.int)
        
        # Resize the image to half its original size
        height, width = img.shape[:factor]
        img_resized = resize(img, (height//factor, width //
                            factor), anti_aliasing=False)
        
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
        return img_resized


class contrast_stretch:
    def __init__(self):
        self.class_name = self.__class__.__name__
        
    def apply(self, img):

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
        return img, self.class_name
    
class gamma_correction:
    def __init__(self, gamma=1.0, alpha=0.0):
        self.class_name = self.__class__.__name__
        self.gamma = gamma
        self.alpha = alpha
        
    def apply(self, img):
        # Convert the image to a numpy array
        img = np.array(self.img).astype(np.int)

        # Normalize the image to the range [0, 1]
        img = img / 255.0
        
        # Apply the gamma correction
        img = np.power(img+self.alpha, self.gamma)

        # Scale the image back to the range [0, 255]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img, self.class_name


class histogram_equalization:
    def __init__(self):
        self.class_name = self.__class__.__name__
    def apply(self, img):
        
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
        return img, self.class_name

# Sharpening is a helpful method for emphasizing the differences between colors in images


#     def unsharp_masking(self, mean, var):
#         noise = NoiseOverlay(self.img).add_gaussian_noise(mean, var)
#         img = np.copy(self.img)
#         sharp = img - noise
#         ImagePlotter(sharp).plot_image_with_histogram(
#             title=f'{self.img_name} {mean}:{var}', cmap='Greys')
#         sharpened = img + sharp
#         ImagePlotter(sharpened).plot_image_with_histogram(
#             title=f'{self.img_name} {mean}:{var}', cmap='Greys')
#         return sharpened
    