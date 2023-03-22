from PIL import Image
import cv2
import numpy as np
from .imageutils import ImagePlotter, ImageUtil


__all__ = ['ColorSpace', 'bgr_to_rgb', 'rgb_to_grayscale', 'inversion', 'rgb_to_HSV']


class ColorSpace:
    def __init__(self, image_path, folder_name, img_name, plot = False, hist = False, save_image = True):
        self.image_path = image_path
        self.folder_name = folder_name
        self.img_name = img_name
        self.hist = hist
        self.plot = plot
        self.save_image = save_image
    
    def process(self, operator):
        
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
    
class ImageColorSpace:
    def __init__(self, img, plot = False, hist = False):
        self.img = img
        self.plot = plot
        self.hist = hist
        
    def process(self, operator):
        
        image_array = np.array(self.img)
        
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
    
    
    
class bgr_to_rgb:
    def __init__(self):
        self.class_name = self.__class__.__name__       
    def apply(self, img):
        # Split the input image into its blue, green, and red channels
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        # Return the blue, green, and red channels as a tuple
        return [r, g, b], self.class_name

class rgb_to_grayscale:
    def __init__(self, ratio=[0.2989, 0.5870, 0.1140]):
        self.class_name = self.__class__.__name__
        self.ratio = ratio

    def apply(self, img):
        gray_image = np.dot(img[..., :3], self.ratio)
        gray_image = np.clip(gray_image, 0, 255).astype(np.int)
        return gray_image, self.class_name

class inversion:
    def __init__(self):
        self.class_name = self.__class__.__name__
    def apply(self, image):
        inverted_image = 255 - image
        return inverted_image, self.class_name

class rgb_to_HSV:
    def __init__(self):
        self.class_name = self.__class__.__name__
    def apply(self, image):
        pil_image = Image.fromarray(image)
        hsv_image = pil_image.convert('HSV')
        # Convert the HSV image back to a NumPy array
        hsv_array = np.array(hsv_image)
        return hsv_array, self.class_name


                
    
    
