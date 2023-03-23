# from tensorly import tenalg
import numpy as np
from .imageutils import ImageUtil, ImagePlotter
from multiprocessing import Pool
from numba import jit
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import inspect
from functools import partial

__all__ = ['ImageProcessor', 'Processor', 'Convolution', 'Dilation', 'Erosion', 'Tilation']

class Processor:
    def __init__(self, image_path, folder_name, img_name, hist = False, plot = False, save_image = True):
        self.image_path = image_path
        self.folder_name = folder_name
        self.img_name = img_name
        self.hist = hist
        self.plot = plot
        self.save_image = save_image
    
    def process(self, operator, kernel_matrix):
        # Open the input image
        image = Image.open(self.image_path)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array, kernel_matrix)
        
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
    
class ImageProcessor:
    def __init__(self, img, plot = False, hist = False):
        self.img = img
        self.hist = hist
        self.plot = plot
    
    def process(self, operator, kernel_matrix):
        # Open the input image
        image_array = np.array(self.img)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array, kernel_matrix)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if self.plot == True:
            if self.hist == True:
                ImagePlotter(output).plot_image_with_histogram(f'{class_name}')
                
            else:
                ImagePlotter(output).plot_image(f'{class_name}')
        
        return output
    
class Convolution:
    def __init__(self):
        self.class_name = self.__class__.__name__
        
    def apply(self, img, kernel_matrix):        
        k_matrix = np.array(kernel_matrix).astype(np.float64)
        
        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2
        
        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant'))
        
        # Normalize the Kernel
        if np.sum(k_matrix) == 0:
            k_matrix_norm = k_matrix
        else:
            k_matrix_norm = np.array(k_matrix) / np.sum(k_matrix)
                    
        output = [self.weighted_arithmetic_mean(padded_image[row-padding_size:row+padding_size+1,
                                                        col-padding_size:col+padding_size+1, channel], k_matrix_norm)
                            for row in range(padding_size, padded_image.shape[0] - padding_size)
                            for col in range(padding_size, padded_image.shape[1] - padding_size)
                            for channel in range(0, img.shape[2])]
        
        output = np.array(output).reshape(img.shape)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output, self.class_name
    
    @staticmethod
    @jit(nopython=True)
    def weighted_arithmetic_mean(sub_image, weight_matrix):
        return np.sum(np.multiply(sub_image, weight_matrix))
    
class HOG:
    def __init__(self, ref_img, threshold):
        self.ref_img = ref_img
        self.threshold = threshold
        self.class_name = self.__class__.__name__
    
    @staticmethod
    def apply(self, target_img):
        
        target_img_hpadding = self.ref_img.shape[0] // 2
        target_img_vpadding = self.ref_img.shape[1] // 2
        
        
        img = np.zeros_like(target_img)
        # nested for-loop version of the code
        for row in range(target_img_vpadding, target_img.shape[0] - target_img_vpadding):
            for col in range(target_img_hpadding, target_img.shape[1] - target_img_hpadding):
                for channel in range(0, target_img.shape[2]):
                    output = self.threshold_detection(target_img[row-target_img_vpadding:row+target_img_vpadding+1,
                                                        col-target_img_hpadding:col+target_img_hpadding+1, channel], self.ref_img, self.threshold)
                    
                    if output == 255:
                        start_point = (row-target_img_hpadding, col-target_img_vpadding)    # X, Y coordinates of top-left corner
                        end_point = (row+target_img_hpadding, col+target_img_vpadding)    # X, Y coordinates of bottom-right corner

                        # Specify the color of the rectangle as BGR values
                        color = (100)       # green
                        
                        # Specify the thickness of the lines used to draw the rectangle
                        thickness = 2
                        
                        # Draw the rectangle on the input image using cv2.rectangle()
                        cv2.rectangle(img, start_point, end_point, color, thickness)
                    
                    img[row, col, channel] = output
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output, self.class_name
    
    @staticmethod
    def threshold_detection(self, target_img, ref_img, threshold):
        if self.histogram_chi_difference(target_img, ref_img) < threshold:
            return 255
        else:
            return 0
    
    @staticmethod
    def histogram_chi_difference(self, target_img, ref_img):
        hist_x = self.hog_histogram(target_img)
        hist_y = self.hog_histogram(ref_img)
        
        num = (hist_x - hist_y) ** 2
        denom = hist_x + hist_y + 1e-6
        diff = 0.5 * np.sum(num / denom)
        return diff
        
    @staticmethod
    def hog_histogram(self, img):
         # Calculate x and y gradients using Sobel operator
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Calculate magnitude and angle of gradients
        _, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Build histogram for angles
        hist, bins = np.histogram(angle, bins=9, range=(0., 180.))
        
        return hist, bins
        
    
class Erosion:
    def __init__(self):
        self.class_name = self.__class__.__name__
        
    @staticmethod
    #@jit(nopython=True)
    def min(sub_image, weight_matrix):
        # Create a flat version of the input array
        flat_arr = sub_image.ravel()

        # Create a flat version of the weight matrix
        flat_weight_matrix = np.array(weight_matrix).ravel()

        nonzero_mask = flat_weight_matrix != 0
        nonzero_pixels = flat_arr[nonzero_mask]
        
        if nonzero_pixels.size == 0:
            return 0
        else:
            return np.min(nonzero_pixels)

    def apply(self, img=np.full((1, 1), 1), kernel_matrix=np.full((3, 3), 1)):
        # If 2d, make 3d
        if (img.ndim == 2):
            img = np.expand_dims(img, axis=2)

        k_matrix = np.array(kernel_matrix)
        img = np.array(np.copy(img))

        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2

        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((
            padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=255))

        


        output = [Erosion.min(padded_image[row-padding_size:row+padding_size+1,
                                                        col-padding_size:col+padding_size+1, channel], k_matrix)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0, img.shape[2])]

        output = np.array(output).reshape(img.shape)

        output = np.clip(output, 0, 255).astype(np.uint8)

        return output, self.class_name
    
class Dilation:
    def __init__(self):
        self.class_name = self.__class__.__name__
        
    @staticmethod
    @jit(nopython=True)
    def max(sub_image, weight_matrix):
        return np.max(np.multiply(sub_image, weight_matrix))

    def apply(self, img=np.full((1, 1), 1), kernel_matrix=np.full((3, 3), 1)):
        # If 2d, make 3d
        if (img.ndim == 2):
            img = np.expand_dims(img, axis=2)

        k_matrix = np.array(kernel_matrix)
        img = np.array(np.copy(img))

        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2

        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((
            padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0))


        output = [self.max(padded_image[row-padding_size:row+padding_size+1,
                                                        col-padding_size:col+padding_size+1, channel], k_matrix)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0, img.shape[2])]

        output = np.array(output).reshape(img.shape)

        output = np.clip(output, 0, 255).astype(np.uint8)

        return output, self.class_name
    
class Trimmed_Median:
    def __init__(self, alpha=0.3):
        frame = inspect.currentframe()
        self.class_name = self.__class__.__name__
        self.alpha = alpha

    def apply(self, img=np.full((1, 1), 1), kernel_matrix=np.full((3, 3), 1)):
        # If 2d, make 3d
        if (img.ndim == 2):
            img = np.expand_dims(img, axis=2)

        k_matrix = np.array(kernel_matrix)
        img = np.array(np.copy(img))

        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2

        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((
            padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=255))

        output = [max(padded_image[row-padding_size:row+padding_size+1,
                                                        col-padding_size:col+padding_size+1, channel], k_matrix, self.alpha)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0, img.shape[2])]

        output = np.array(output).reshape(img.shape)

        output = np.clip(output, 0, 255).astype(np.uint8)

        return output, self.class_name
    
    @jit(nopython=True)
    def weighted_alpha_trim_mean(sub_image, weight_matrix, alpha):
        # Shave an alpha ratio of the sorted pixels from the left most and right most indexes and return the average of the remaining indexes
        # Alpha 0.5 = median, Alpha 0 = mean

        # Create a flat version of the input array
        flat_arr = np.array(sub_image).flatten()

        # Create a flat version of the weight matrix
        flat_weight_matrix = np.array(weight_matrix).flatten()

        # Repeat each pixel in the flat array according to the weight matrix
        repeated_pixels = np.repeat(flat_arr, flat_weight_matrix)

        # Sort the repeated pixels
        sorted_pixels = np.sort(repeated_pixels)

        # Determine the left and right indexes that define the desired range.
        left_index = int(alpha * len(sorted_pixels))
        right_index = int((1 - alpha) * len(sorted_pixels))

        return np.mean(sorted_pixels[left_index:right_index])

class Tilation(ImageUtil):
    # Takes Pillow Objects
    def __init__(self, img=np.full((5, 5), 1), name='default', hist=True):
        self.img = img
        self.section_dict = None
        self.image_name = name
        self.hist = hist
        
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)

    def split_image_nxn_sections(self, sections):

        # Get the size of the image
        height, width = self.img.shape[:2]

        layers = 0

        if len(self.img.shape) == 2:
            layers = 1
        else:
            layers = self.img.shape[2]

        # Calculate the height of each section
        section_height = int(np.ceil(height / sections))

        # Calculate the width of each section
        section_width = int(np.ceil(width / sections))

        # Initialize the list to store the sections
        section_list = []

        if layers == 1:
            # Split the image into sections
            for row in range(0, height, section_height):
                for col in range(0, width, section_width):
                    section = self.img[row:row + section_height,
                                       col:col + section_width]
                    section_list.append(section)
        else:
            # Split the image into sections
            for row in range(0, height, section_height):
                for col in range(0, width, section_width):
                    section = self.img[row:row + section_height, col:col + section_width, :]
                    section_list.append(section)

        # Return the output wrapped in a dictionary
        section_dict = {
            'section_height': section_height,
            'section_width': section_width,
            'height': height,
            'width': width,
        }
        self.section_dict = section_dict
        return section_list, section_dict

    def merge_sections_into_image(self, section_list, section_dict):
        # Get the number of channels from the first section
        num_channels = section_list[0].shape[2]

        # Initialize the result image with the correct number of channels
        result_img = np.zeros(
            (section_dict['height'], section_dict['width'], num_channels), dtype=np.uint8)

        # Merge the sections into a single image
        index = 0
        for row in range(0, section_dict['height'], section_dict['section_height']):
            for col in range(0, section_dict['width'], section_dict['section_width']):
                section = section_dict['section_list'][index]
                result_img[row:row + section_dict['section_height'],
                           col:col + section_dict['section_width'], :] = section
                index += 1

        if self.hist == True:
            ImagePlotter(result_img).plot_image_with_histogram(
                title=f'{self.image_name}')
        else:
            ImagePlotter(result_img).plot_image(
                title=f'{self.image_name}')
        return result_img

    def func_pass(x): return x

    def apply_function_nxn_sections(self, func1=func_pass, func2=func_pass, func3=func_pass, *args):

        L, A, B = cv2.split(self.section_dict['section_list'])

        L = [func1(section[:, :, 0], *args)
             for section in self.section_dict['section_list']]
        A = [func2(section[:, :, 1], *args)
             for section in self.section_dict['section_list']]
        B = [func3(section[:, :, 2], *args)
             for section in self.section_dict['section_list']]

        # Apply the functions to each section
        results = [function(section, *args)
                   for section in self.section_dict['section_list']]

        # Return the output
        return {
            'section_list': results,
            'section_height': self.section_dict['section_height'],
            'section_width': self.section_dict['section_width'],
            'height': self.section_dict['height'],
            'width': self.section_dict['width'],
        }

    def show_image_sections(self):
        # Calculate the number of rows and columns for the plot
        n_rows = int(np.sqrt(len(self.section_dict['section_list'])))
        n_cols = int(np.ceil(len(self.section_dict['section_list']) / n_rows))

        # Create a figure with subplots
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        ax = ax.ravel()

        # Plot each section in its own subplot
        for i, section in enumerate(self.section_dict['section_list']):
            ax[i].imshow(section)
            ax[i].axis('off')

        plt.tight_layout()
        plt.show()