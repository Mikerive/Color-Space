import numpy as np
from .imageutils import ImageUtil, ImagePlotter
from multiprocessing import Pool
from functools import partial
from numba import jit
import cv2
import matplotlib.pyplot as plt
from .qualitymeasures import QualityMeasures

counter = 0

class Convolution:
    def __init__(self, img=np.full((1, 1), 1), img_name='default', hist=False):
         
        self.img = np.array(img)
        self.list_img = img
            
        self.img_name = img_name
        
        self.hist = hist
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
            
    default_matrix = np.full((3, 3), 1)

    @staticmethod
    def kernel_operation(index, params):
        padded_img = params['img_padded']
        padding_size = params['padding_size']
        func = params['func']
        kernel_matrix = params['kernel_matrix']
        args = params['args']

        row, col, channel = index
        # Apply the parameter function to the sub_image
        # Apply the value to the image
        return func(padded_img[row+padding_size, col+padding_size, channel], kernel_matrix, *args)
        
    def sliding_kernel(self, kernel_matrix, func, *args, num_processes=2):
        
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
        
        k_matrix = np.array(kernel_matrix).astype(np.float64)
        img = np.array(np.copy(self.img))
        
        print(img.shape)
        
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2
        
        # Get the coordinates of the center pixel in the kernel matrix
        
        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant'))
        
        if np.sum(k_matrix) == 0:
            k_matrix_norm = k_matrix
        else:
            k_matrix_norm = np.array(k_matrix) / np.sum(k_matrix)
        
        
        
        # working
        print('1')
        output = np.zeros_like(img)
        output = [func(padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel], k_matrix_norm, *args)
                                            for row in range(padding_size, padded_image.shape[0] - padding_size)
                                            for col in range(padding_size, padded_image.shape[1] - padding_size)
                                            for channel in range(img.shape[2])]
        print('2')
        
        # # Define the kernel operation parameters
        # kernel_operation = Convolution().kernel_operation
        # kernel_func = partial(
        #     kernel_operation,
        #     params={
        #         'img_padded': padded_image,
        #         'padding_size': padding_size,
        #         'kernel_matrix': k_matrix_norm,
        #         'func': func,
        #         'args': args
        #     }
        # )

        # output = np.zeros_like(img)
        # index_tuples = [(row, col, channel) for (row, col, channel) in np.ndindex(img.shape)]
        # with Pool(processes=num_processes) as pool:
        #     async_results = pool.map_async(kernel_func, index_tuples)
        # output = np.array(list(async_results.get())).reshape(img.shape)
            
        # print(output.shape)
        
        print('1: ', np.array(output).shape)
        
        output = np.array(output).reshape(img.shape)
        
        print('2: ', output.shape)
        
        print(QualityMeasures(self.img).histogram_distance_euclidian(output))
        
        plt.imshow(output)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(
                title=f'{self.img_name}_n={kernel_size}')
        else:
            ImagePlotter(output).plot_image(
                title=f'{self.img_name}_n={kernel_size}')

        if img.shape[2] == 1:
            output = np.squeeze(output)
            path = ImageUtil(output).save_image_to_folder(
                'Image/Convolution/', f"{self.img_name}.png")
            return output, path
        else:
            path = ImageUtil(output).save_image_to_folder(
                'Image/Convolution/', f"{self.img_name}.png")
            return output, path

    @staticmethod
    @jit(nopython=False)
    def weighted_arithmetic_mean(sub_image, weight_matrix=default_matrix):
        # print('sub_image', np.array(sub_image).shape)
        # print('weight_matrix', weight_matrix.shape)
        
        product = np.multiply(sub_image, weight_matrix)

        return np.array(np.sum(product) / (product.shape[0] * product.shape[1]))

    def gaussian_kernel(n, sigma):
        x, y = np.meshgrid(np.arange(-n // 2 + 1, n // 2 + 1), np.arange(-n // 2 + 1, n // 2 + 1))
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g

    def weighted_geometric_mean(self, arr, weight_matrix=default_matrix):
        # Normalize the weight matrix
        weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)

        # Compute the weighted product of the neighborhood values
        weighted_product = np.prod(arr ** weight_matrix)

        # Compute the weighted geometric mean
        weighted_geometric_mean = np.power(weighted_product, 1.0 / np.sum(weight_matrix))

        return weighted_geometric_mean

    def weighted_median(self, arr, weight_matrix = default_matrix):
        
        # Create a flat version of the input array
        flat_arr = arr.flatten()

        # Create a flat version of the weight matrix
        flat_weight_matrix = np.array(weight_matrix).flatten()

        # Repeat each pixel in the flat array according to the weight matrix
        repeated_pixels = np.repeat(flat_arr, flat_weight_matrix)

        # Sort the repeated pixels
        sorted_pixels = np.sort(repeated_pixels)
                
        return np.median(sorted_pixels)

    def max(self, arr, weight_matrix=default_matrix):
        return np.max(arr)

    def min(self, arr, weight_matrix=default_matrix):
        return np.min(arr)

    def midpoint(self, arr, weight_matrix=default_matrix):
        return (np.max(arr) + np.min(arr))/2

    def weighted_alpha_trim_mean(self, arr, weight_matrix=default_matrix, alpha=0.3):
        # Shave an alpha ratio of the sorted pixels from the left most and right most indexes and return the average of the remaining indexes
        # Alpha 0.5 = median, Alpha 0 = mean
        
        # Create a flat version of the input array
        flat_arr = arr.flatten()

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
