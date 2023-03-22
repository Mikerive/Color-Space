import numpy as np
from skimage.transform import resize
from PIL import Image

from .imageutils import *

__all__ = ['Filters', 'ImageFilters' , 'contrast_stretch', 'gamma_correction', 'histogram_equalization', 'subimage']

class Filters:
    def __init__(self, image_path, folder_name, img_name, plot = False, hist = False, save_image = True):
        self.image_path = image_path
        self.folder_name = folder_name
        self.img_name = img_name
        self.hist = hist
        self.plot = plot
        self.save_image = save_image
    
    def process(self, operator):
        # Open the input image
        image = Image.open(self.image_path)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
            
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if self.plot == True:
            if self.hist == True:
                ImagePlotter(output).plot_image_with_histogram(f'{self.img_name}_{class_name}')
                
            else:
                ImagePlotter(output).plot_image(f'{self.img_name}_{class_name}')
        
        if self.save_image == True:
            path = ImageUtil(output).save_image_to_folder(
                f'Image/{self.folder_name}/', f"{self.img_name}.png")
            return output, path
        
        return output

class ImageFilters:
    def __init__(self, img, plot = False, hist = False):
        self.img = img
        self.plot = plot
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
        
        if self.plot == True:
            if self.hist == True:
                ImagePlotter(output).plot_image_with_histogram(f'{self.img_name}_{class_name}')
                
            else:
                ImagePlotter(output).plot_image(f'{self.img_name}_{class_name}')
        
        return output


class subimage:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.class_name = self.__class__.__name__
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
    
    def apply(self, img):
        img = np.asarray(img)
        
        if np.ndim(img)==2:
            img = np.squeeze(img)
            subimg = img[self.y_start:self.y_end, self.x_start:self.x_end]
            return subimg, self.class_name
        else:
            # Create the subimage
            subimg = img[self.y_start:self.y_end, self.x_start:self.x_end, :]
            return subimg, self.class_name

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
        img = np.array(img).astype(np.int)

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
        img = np.array(img).astype(np.int8)
        
        # Calculate the cdf
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()

        # Calculate the normalized cdf
        cdf_normalized = cdf / cdf.max()

        # Map the pixel values to the normalized cumulative distribution
        img = np.interp(img, bins[:-1], cdf_normalized)

        # Convert the image back to 8-bit unsigned integer format
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        
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
    