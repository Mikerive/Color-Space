# from tensorly import tenalg
import numpy as np
from .imageutils import ImageUtil, ImagePlotter
from multiprocessing import Pool
from functools import partial
from numba import jit
import numba as nb
import cv2
import matplotlib.pyplot as plt
from .qualitymeasures import QualityMeasures
from .imageutils import *

counter = 0

class Convolution:
    def __init__(self, img=np.full((1, 1), 1), folder_name = 'default', img_name='default', hist=False):
         
        self.img = np.array(img)
        self.list_img = img
            
        self.img_name = img_name
        self.folder_name = folder_name
        
        self.hist = hist
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
            
    default_matrix = np.full((3, 3), 1)

    # @staticmethod
    # def kernel_operation(index, params):
    #     padded_img = params['img_padded']
    #     padding_size = params['padding_size']
    #     func = params['func']
    #     kernel_matrix = params['kernel_matrix']
    #     args = params['args']

    #     row, col, channel = index
    #     # Apply the parameter function to the sub_image
    #     # Apply the value to the image
    #     return func(padded_img[row+padding_size, col+padding_size, channel], kernel_matrix, *args)
    
    # def threshold_kernel_operation(self, kernel_matrix, func, threshold = 5):
        
    #     # If 2d, make 3d
    #     if (self.img.ndim == 2):
    #         self.img = np.expand_dims(self.img, axis=2)

    #     k_matrix = np.array(kernel_matrix).astype(np.float64)
    #     img = np.array(np.copy(self.img))
        
    #     height, width, channels = np.array(img).shape
            
    #     # If kmatrix sum is 0, don't divide by zero
    #     k_matrix_norm = np.where(np.sum(k_matrix) == 0, k_matrix, np.array(k_matrix) / np.sum(k_matrix))
        
    #     output_tensor = tenalg.multi_mode_dot(img, [k_matrix_norm]*channels, modes=[0, 1])

    #     output_tensor = np.where(output_tensor > threshold, 255, 0)

    #     output = np.squeeze(np.array(output_tensor).reshape(-1, width, channels)).astype(np.uint8)
        
    #     print('2: ', output.shape)

    #     print(QualityMeasures(self.img).histogram_distance_euclidian(output))

    #     plt.imshow(output)

    #     output = np.clip(output, 0, 255).astype(np.uint8)

    #     if self.hist == True:
    #         ImagePlotter(output).plot_image_with_histogram(
    #             title=f'{self.img_name}_n={k_matrix.shape[0]}')
    #     else:
    #         ImagePlotter(output).plot_image(
    #             title=f'{self.img_name}_n={k_matrix.shape[0]}')

    #     if img.shape[2] == 1:
    #         output = np.squeeze(output)
    #         path = ImageUtil(output).save_image_to_folder(
    #             'Image/Threshold/', f"{self.img_name}.png")
    #         return output, path
    #     else:
    #         path = ImageUtil(output).save_image_to_folder(
    #             'Image/Threshold/', f"{self.img_name}.png")
    #         return output, path

        
        
    def convolution(self, kernel_matrix,**kwargs):
        
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
        
        
        k_matrix = np.array(kernel_matrix).astype(np.float64)
        img = np.array(np.copy(self.img))
        
        # Get dimensions of the kernel.
        kernel_size = k_matrix.shape[0]
        padding_size = kernel_size // 2

        # Get the coordinates of the center pixel in the kernel matrix
        
        # Pad the image with zeros
        padded_image = np.array(np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant'))
        
        if np.sum(k_matrix) == 0:
            k_matrix_norm = k_matrix
        else:
            k_matrix_norm = np.array(k_matrix) / np.sum(k_matrix)

        # output = np.zeros_like(img)
        # index_tuples = [(row, col, channel) for (row, col, channel) in np.ndindex(img.shape)]
        # with Pool(processes=num_processes) as pool:
        #     async_results = pool.map_async(kernel_func, index_tuples)
        # output = np.array(list(async_results.get())).reshape(img.shape)
            
        # print(output.shape)
        
        # List Comprehension Based Approach
        
        # Define the default kernel operation parameters
        # kernel_func = partial(func,
        #                       #threshold=kwargs.get('threshold', 3),
        #                       weight_matrix=k_matrix_norm
        #                       )

        # output = np.zeros_like(img)
        # 
        
        # @staticmethod
        @jit(nopython=True)
        def weighted_arithmetic_mean(sub_image, weight_matrix):
            product = np.multiply(sub_image, weight_matrix)
            # mean = np.array(np.sum(product) / np.sum(weight_matrix))
            return np.sum(product)
        
        # output = np.zeros_like(img)
        # for row in range(padding_size, padded_image.shape[0] - padding_size-1):
        #     for col in range(padding_size, padded_image.shape[1] - padding_size-1):
        #         for channel in range(0, img.shape[2]):
        #             sub_img = padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel]
        #             output[row,col,channel] = weighted_arithmetic_mean(sub_img, k_matrix_norm)
                    
        output = [weighted_arithmetic_mean(padded_image[row-padding_size:row+padding_size+1,
                                                        col-padding_size:col+padding_size+1, channel], k_matrix_norm)
                            for row in range(padding_size, padded_image.shape[0] - padding_size)
                            for col in range(padding_size, padded_image.shape[1] - padding_size)
                            for channel in range(0, img.shape[2])]
                    
            
        
        output = np.array(output).reshape(img.shape)
        
        # print(QualityMeasures(self.img).histogram_distance_euclidian(output))
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(
                title=f'{self.img_name}_n={kernel_size}')
        else:
            ImagePlotter(output).plot_image(
                title=f'{self.img_name}_n={kernel_size}')

        if img.shape[2] == 1:
            path = ImageUtil(output).save_image_to_folder(
                f'Image/{self.folder_name}/', f"{self.img_name}.png")
            return output, path
        else:
            path = ImageUtil(output).save_image_to_folder(
                f'Image/{self.folder_name}/', f"{self.img_name}.png")
            return output, path

    
    
    @staticmethod
    @jit(nopython=True)
    def weighted_arithmetic_mean_threshold(self, sub_image, weight_matrix=default_matrix, threshold=3):

        product = np.multiply(sub_image, weight_matrix)
        
        val = np.array(np.sum(product) / (product.shape[0] * product.shape[1]))
        if val > threshold:
            return 255
        else:
            return 0
        
    # @staticmethod
    # @jit(nopython=True)
    # def weighted_geometric_mean(self, arr, weight_matrix=default_matrix):
    #     # Normalize the weight matrix
    #     weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)

    #     # Compute the weighted product of the neighborhood values
    #     weighted_product = np.prod(arr ** weight_matrix)

    #     # Compute the weighted geometric mean
    #     weighted_geometric_mean = np.power(weighted_product, 1.0 / np.sum(weight_matrix))

    #     return weighted_geometric_mean
    
    # @staticmethod
    # @jit(nopython=True)
    # def weighted_median(self, arr, weight_matrix = default_matrix):
        
    #     # Create a flat version of the input array
    #     flat_arr = arr.flatten()

    #     # Create a flat version of the weight matrix
    #     flat_weight_matrix = np.array(weight_matrix).flatten()

    #     # Repeat each pixel in the flat array according to the weight matrix
    #     repeated_pixels = np.repeat(flat_arr, flat_weight_matrix)

    #     # Sort the repeated pixels
    #     sorted_pixels = np.sort(repeated_pixels)
                
    #     return np.median(sorted_pixels)


    # @staticmethod   
    # @jit(nopython=True)
    # def max(self, arr, weight_matrix=default_matrix):
    #     return np.max(arr)


    # @staticmethod
    # @jit(nopython=True)
    # def min(self, arr, weight_matrix=default_matrix):
    #     return np.min(arr)


    # @staticmethod
    # @jit(nopython=True)
    # def midpoint(self, arr, weight_matrix=default_matrix):
    #     return (np.max(arr) + np.min(arr))/2


    # @staticmethod
    # @jit(nopython=True)
    # def weighted_alpha_trim_mean(self, arr, weight_matrix=default_matrix, alpha=0.3):
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

        @jit(nopython=True)
        def magnitude(mag, himg, vimg):
            for i in range(himg.shape[0]):
                for j in range(himg.shape[1]):
                    mag[i][j] = (himg[i][j]**2 + vimg[i][j]**2)**0.5
            return (mag / np.amax(mag)) * 255.0
            
        return magnitude(mag,himg,vimg).astype(np.uint8)
    
    def gradient_direction(self, himg, vimg):
        angle = np.arctan2(vimg.astype(np.float32),
                           himg.astype(np.float32))  # compute angle

        # Convert the radians to degrees and scale the range from [0, pi] to [0, 255]
        return ((angle + np.pi) * 255.0 / (2 * np.pi)).astype(np.uint8)
        
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
    

class Segment:
    def __init__(self, img=np.full((10, 10), 1), name='default', hist=False):
        self.img = np.array(img).astype(np.int32)
        self.img_name = name
        self.hist = hist

        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)

    def segment_image(self, distance=10, width=8):
        # Calculate the histogram of the image
        hist, bins = np.histogram(self.img.flatten(), bins=256, range=(0, 255))

        # Find the peaks in the histogram using the find_peaks function from SciPy
        from scipy.signal import find_peaks

        # Find peaks with a minimum height of 1/4 of the maximum value
        minima, _ = find_peaks(-hist, distance=distance, width=width)

        # Find the ranges of pixel values corresponding to halfway between each peak
        ranges = []
        for i in range(len(minima)):
            if i == 0:
                start = 0
            else:
                start = minima[i]

            # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
            end = minima[i+1] if i < len(minima) - 1 else 255
            ranges.append((start, end))

        # Create a binary mask for each range of pixel values
        masks = []
        for r in ranges:
            # Values within the range arr assigned 255, the rest 0
            mask = np.zeros_like(self.img)
            mask[(self.img >= r[0]) & (self.img <= r[1])] = 255
            masks.append(mask)

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image_with_histogram(
                title=f'mask {ranges[i][0]}:{ranges[i][1]}')

        return masks

    def global_threshold(self, mode='mean', deltaT=3):
        """
        Applies global thresholding to an input grayscale image.
        
        :param image: A grayscale image as a NumPy array.
        :param threshold: The threshold value.
        :return: The thresholded image as a binary NumPy array.
        """
        image = self.img.astype(int)
        threshold = None

        if mode == 'mean':
            threshold = (np.max(image)+np.min(image))//2
        elif mode == 'median':
            threshold = np.median(image)
        # Minimizes the intra-class variance of the two resulting classes (foreground and background)
        elif mode == 'otsu':
            _, output = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return output

        # If T is within three pixels of the last T, update and terminate while loop
        done = False
        while done == False:
            img1, img2 = np.zeros_like(image), np.zeros_like(image)
            img1, img2 = image[image < threshold], image[image > threshold]
            thresholdnext = (np.mean(img1)+np.mean(img2))//2
            if abs(thresholdnext-threshold) < deltaT:
                done = True
            threshold = thresholdnext

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Create a binary mask by comparing the image with the threshold value
        mask1, mask2 = np.zeros_like(image), np.zeros_like(image)

        mask1[image < threshold] = 255
        mask2[image >= threshold] = 255

        ImagePlotter(mask1).plot_image_with_histogram(
            title=f'mask 0:{threshold}')
        ImagePlotter(mask2).plot_image_with_histogram(
            title=f'mask {threshold}:255')
        # Clip the output image to ensure that pixel values are within [0, 255]
        output = np.clip(mask1, 0, 255).astype(np.uint8)
        output = np.clip(mask2, 0, 255).astype(np.uint8)

        return mask1, mask2

    # def get_window(self, img, row, col, window_size):
    #     # When iterating through an image, we need to know the windows.
    #     height, width, depth = img.shape

    #     row_min = max(0, row - window_size // 2)
    #     row_max = min(height, row + window_size // 2 + 1)
    #     col_min = max(0, col - window_size // 2)
    #     col_max = min(width, col + window_size // 2 + 1)

        return img[row_min:row_max, col_min:col_max, :]

    def Pixel_Filter(self, window_size, func_type = 'Sauvola', **kwargs):
        # Takes Niblack, Sauvola, and Bernsen filter functions
        
        img = np.copy(self.img)
        # Calculate the padding size based on the window size
        padding_size = window_size // 2

        # Pad the image with zeros
        padded_image = np.pad(img, ((padding_size, padding_size),
                              (padding_size, padding_size), (0, 0)), mode='constant')
        
        print(padded_image.shape)
        
        # filter_func = partial(func,
        #         k=kwargs.get('k'),
        #         R=kwargs.get('R')
        #         )
        
        if func_type == 'Niblack':
            func = Niblack
        elif func_type == 'Sauvola':
            func = Sauvola
        else:
            func = Bernsen
        
        @jit(nopython=True)
        def Niblack(self, window, row, col, channel, k=-0.2):
            # Compute the mean and standard deviation for each channel separately
            means = np.mean(window)
            stds = np.std(window)

            thresholds = means + k * stds

            if window[row, col, channel] > thresholds:
                return 255
            else:
                return 0

        @jit(nopython=True)
        def Sauvola(self, window, row, col, channel, k=0.34, R=128):
            # Compute the mean and standard deviation for each channel separately
            means = np.mean(window)
            stds = np.std(window)

            # Compute the local threshold for each channel
            thresholds = means * (1.0 + k * (-1 + stds / R))

            if window[row, col, channel] > thresholds:
                return 255
            else:
                return 0

        @jit(nopython=True)
        def Bernsen(self, window, row, col, channel):
            # Compute the mean and standard deviation for each channel separately
            maxs = np.max(window['img_window'])
            mins = np.min(window['img_window'])

            thresholds = (maxs + mins)/2
            if window[row, col, channel] > thresholds:
                return 255
            else:
                return 0

        output = np.zeros_like(img)
        output = [func(padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel], row, col, channel)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0,img.shape[2])]
        
        output = np.clip(output, 0, 255).astype(np.uint8)

        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(
                title=f'{self.img_name}_{window_size}')
        else:
            ImagePlotter(output).plot_image(
                title=f'{self.img_name}')

        return output
    
    

    def global_multiple_threshold(self, minima):

        # Find the ranges of pixel values corresponding to halfway between each peak
        ranges = []
        for i in range(len(minima)):
            if i == 0:
                start = 0
            else:
                start = minima[i]

            # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
            end = minima[i+1] if i < len(minima) - 1 else 255
            ranges.append((start, end))

        # Create a binary mask for each range of pixel values
        masks = []
        for r in ranges:
            # Values within the range arr assigned 255, the rest 0
            mask = np.zeros_like(self.img)
            mask[(self.img >= r[0]) & (self.img <= r[1])] = 255
            masks.append(mask)

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image_with_histogram(
                title=f'Map {ranges[i][0]}:{ranges[i][1]}')

        return masks

    def adaptive_threshold_segmentation(self, n=30, background_difference=5, deltaT=3):

        image = self.img.astype(int)

        image_dict = Tilation(image).split_image_nxn_sections(n)
        for i, image in enumerate(image_dict['section_list']):
            for layer in range(image.shape[2]):

                img_layer = image[:, :, layer]

                # Don't Segment if background
                if np.max(image) - np.min(image) < background_difference:
                    image_dict['section_list'][i][:, :,
                                                  layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = (np.max(image)+np.min(image))//2
                    # If T is within three pixels of the last T, update and terminate while loop
                    done = False
                    while done == False:
                        img1, img2 = np.zeros_like(image), np.zeros_like(image)
                        img1, img2 = image[image <
                                           threshold], image[image > threshold]
                        thresholdnext = (np.mean(img1)+np.mean(img2))//2
                        if abs(thresholdnext-threshold) < deltaT:
                            done = True
                        threshold = thresholdnext

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255
                    image_dict['section_list'][i][:, :, layer] = np.clip(
                        mask, 0, 255).astype(np.uint8)

        Tilation(name=f'adaptive_seg_{self.img_name} {n}:{background_difference}').merge_sections_into_image(
            image_dict)
        return image



# class LinearRegression:
    # https://medium.com/analytics-vidhya/linear-regression-from-scratch-in-python-b6501f91c82d
    # def __init__(self, lr=0.01, deltaW=0.1):
    #     self.lr = lr
    #     self.epochs = epochs
    #     self.weights = None
    #     self.bias = None
        
    # # Training function: fit
    # def fit(self, X, y):
    #     # shape of X: (number of training examples: m, number of
    #     # features: n)
    #     m, n = X.shape

    #     # Initializing weights as a matrix of zeros of size: (number
    #     # of features: n, 1) and bias as 0
    #     self.weights = np.zeros((n, 1))
    #     self.bias = 0

    #     # reshaping y as (m,1) in case your dataset initialized as
    #     # (m,) which can cause problems
    #     y = y.reshape(m, 1)

    #     # empty lsit to store losses so we can plot them later
    #     # against epochs
    #     losses = []

    #     # Gradient Descent loop/ Training loop
    #     for epoch in range(self.epochs):

    #         # Calculating prediction: y_hat or h(x)
    #         y_hat = np.dot(X, self.weights) + self.bias

    #         # Calculting loss
    #         loss = np.mean((y_hat - y)**2)

    #         # Appending loss in list: losses
    #         losses.append(loss)

    #         # Calculating derivatives of parameters(weights, and
    #         # bias)
    #         dw = (1/m)*np.dot(X.T, (y_hat - y))
    #         db = (1/m)*np.sum((y_hat - y))
    #         # Updating the parameters: parameter := parameter - lr*derivative
    #         # of loss/cost w.r.t parameter)
    #         self.weights -= self.lr*dw
    #         self.bias -= self.lr*db

    #     # returning the parameter so we can look at them later
    #     return self.weights, self.bias, losses
