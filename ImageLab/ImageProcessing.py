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

__all__ = ['ImageProcessor', 'Convolution', 'Dilation', 'Erosion', 'EdgeDetect', 'Tilation', 'Kernels', 'Segmentation_Filter']

class ImageProcessor:
    def __init__(self, image_path, folder_name, img_name, hist = False):
        self.image_path = image_path
        self.folder_name = folder_name
        self.img_name = img_name
        self.hist = hist
    
    def process(self, operator, kernel_matrix):
        # Open the input image
        image = Image.open(self.image_path)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
        
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array, kernel_matrix)

        path = ImageUtil(output).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{self.img_name}_{class_name}.png")
        
        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(f'{self.img_name}_{class_name}')
            
        else:
            ImagePlotter(output).plot_image(f'{self.img_name}_{class_name}')
        
        return output, path
    
class Convolution:
    def __init__(self):
        self.class_name = self.__class__.__name__
        
    def apply(self, img=np.full((1, 1), 1), kernel_matrix=np.full((3, 3), 1)):
        # If 2d, make 3d
        if (img.ndim == 2):
            img = np.expand_dims(img, axis=2)
        
        k_matrix = np.array(kernel_matrix).astype(np.float64)
        img = np.array(np.copy(img))
        
        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2
        
        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant'))
        
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
            padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0))

        


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

class Segmentation_Filter:
    def __init__(self, alpha=0.3, k=0.34, R=128):
        self.class_name = self.__class__.__name__
        self.alpha = alpha
        self.k = k
        self.R = R
    
    @staticmethod
    @jit(nopython=True)
    def Niblack(window, center_index, k=-0.2):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window)
        stds = np.std(window)

        thresholds = means + k * stds

        if window[center_index, center_index] > thresholds:
            return 255
        else:
            return 0

    @staticmethod
    @jit(nopython=True)
    def Sauvola(window, center_index, k=0.34, R=128):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window)
        stds = np.std(window)

        # Compute the local threshold for each channel
        thresholds = means * (1.0 + k * (-1 + stds / R))

        if window[center_index, center_index] > thresholds:
            return 255
        else:
            return 0
        
    @staticmethod
    @jit(nopython=True)
    def Bernsen(window, center_index):
        # Compute the mean and standard deviation for each channel separately
        maxs = np.max(window['img_window'])
        mins = np.min(window['img_window'])

        thresholds = (maxs + mins)/2
        if window[center_index, center_index] > thresholds:
            return 255
        else:
            return 0

    def apply(self, img, window_size, func_type='Sauvola'):
        # Takes Niblack, Sauvola, and Bernsen filter functions

        img = np.copy(img)
        # Calculate the padding size based on the window size
        padding_size = window_size // 2

        # Pad the image with zeros
        padded_image = np.pad(img, ((padding_size, padding_size),
                                    (padding_size, padding_size), (0, 0)), mode='constant')

        if func_type == 'Niblack':
            func = partial(Segmentation_Filter.Niblack, k = self.k)
        elif func_type == 'Sauvola':
            func = partial(Segmentation_Filter.Sauvola, k=self.k, R=self.R)
        else:
            func = partial(Segmentation_Filter.Bernsen)

        output = np.zeros_like(img)
        output = [func(padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel], padding_size)
                    for row in range(padding_size, padded_image.shape[0] - padding_size)
                    for col in range(padding_size, padded_image.shape[1] - padding_size)
                    for channel in range(0, img.shape[2])]
        
        output = np.array(output).reshape(img.shape)

        output = np.clip(output, 0, 255).astype(np.uint8)

        return output, self.class_name







class EdgeDetect:
    def __init__(self, img=np.full((1, 1), 1), folder_name='default', img_name='default', hist=False):
        self.img = np.array(img)
        self.list_img = img

        self.img_name = img_name
        self.folder_name = folder_name

        self.hist = hist
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)

    def prewitt_edge_detector(self, n, threshold=4):
        v_kernel = self.prewitt_kernel(n)
        h_kernel = v_kernel.T
        
        vimg, _ = Convolution(self.img, f'prewitt_v_n={n}').convolution(v_kernel, Convolution().weighted_arithmetic_mean_threshold, threshold = threshold)
        himg, _ = Convolution(self.img, f'prewitt_h_n={n}').convolution(h_kernel, Convolution().weighted_arithmetic_mean_threshold, threshold = threshold)
        
        magnitude = self.gradient_magnitude(himg, vimg)
        direction = self.gradient_direction(himg, vimg)
        
        if self.hist == True:
            ImagePlotter(magnitude).plot_image_with_histogram(
                title=f'{self.img_name}_n={n}')
            ImagePlotter(direction).plot_image(
                title=f'{self.img_name}_n={n}')
        else:
            ImagePlotter(magnitude).plot_image(
                title=f'{self.img_name}_n={n}')
            ImagePlotter(direction).plot_image(
                title=f'{self.img_name}_n={n}')

        _ = ImageUtil(magnitude).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{self.img_name}_magnitude.png")
        _ = ImageUtil(direction).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{self.img_name}_direction.png")
        return magnitude, direction
        
    @staticmethod
    def gradient_mag(himg, vimg):
        mag = np.zeros_like(himg).astype(np.float64)
        return EdgeDetect.magnitude(mag,himg,vimg).astype(np.uint8)
    
    @staticmethod
    @jit(nopython=True)
    def magnitude(mag, himg, vimg):
        for i in range(himg.shape[0]):
            for j in range(himg.shape[1]):
                mag[i][j] = (himg[i][j]**2 + vimg[i][j]**2)**0.5
        return (mag / np.amax(mag)) * 255.0
    
    def gradient_direction(self, himg, vimg):
        angle = np.arctan2(vimg.astype(np.float32),
                           himg.astype(np.float32))  # compute angle

        # Convert the radians to degrees and scale the range from [0, pi] to [0, 255]
        return ((angle + np.pi) * 255.0 / (2 * np.pi)).astype(np.uint8)
        
    def hough_transform(img_bin, theta_res=1, rho_res=1):
        h, w = img_bin.shape
        diag_len = int(np.ceil(np.sqrt(h*h + w*w)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2 / rho_res + 1)
        thetas = np.arange(0, 180, theta_res)

        cos_t = np.cos(np.deg2rad(thetas))
        sin_t = np.sin(np.deg2rad(thetas))
        num_thetas = len(thetas)

        accumulator = np.zeros(
            (int(2 * diag_len / rho_res), num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img_bin)

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                rho = int((x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
                accumulator[rho, t_idx] += 1

        return accumulator, thetas, rhos

class Kernels():
    def __init__(self):
        pass
    def prewitt_kernel(self, size):
        """
        Generates the Vertical Prewitt Kernel of specified size
        """
        # Create the Prewitt matrix.
        kernel = np.zeros((size, size))

        for j in range(size):
            if j < size // 2:
                kernel[:, j] = -1
            elif j > size // 2:
                kernel[:, j] = 1

        return kernel

    def gaussian_kernel(n, sigma):
        x, y = np.meshgrid(np.arange(-n // 2 + 1, n // 2 + 1),
                           np.arange(-n // 2 + 1, n // 2 + 1))
        g = np.exp(-(x**2 + y**2) // (2 * sigma**2))
        return g

    def LoG_kernel(n, sigma):

        # Create a 1-D array of indices, centering them around 0
        ind = np.arange(-n // 2, n // 2 + 1)

        # Create 2D arrays of zeros for the LoG filter and Gaussian filter
        L, G = np.meshgrid(ind, ind, indexing='ij')
        LoG_filter = (-1 / (np.pi * sigma ** 4)) * (1 - (L ** 2 + G ** 2) /
                                                    (2 * sigma ** 2)) * np.exp(-(L ** 2 + G ** 2) / (2 * sigma ** 2))

        # Normalize the filter so that its values sum up to 0
        return LoG_filter / np.sum(LoG_filter)

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