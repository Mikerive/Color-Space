import numpy as np
from skimage.transform import resize
from PIL import Image
import cv2
from skimage.feature import hog
from numba import njit, jit, prange
import numpy as np
from numba import prange
from .imageutils import *
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

__all__ = ['Filters', 'ImageFilters', 'HOG', 'HOG_Tilation', 'HOG_Parallel', 'contrast_stretch', 'gamma_correction', 'histogram_equalization', 'subimage']

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
        image_array = np.array(self.img)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)
            
        # Apply the operator to the input image
        output, class_name = operator.apply(image_array)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if self.plot == True:
            if self.hist == True:
                ImagePlotter(output).plot_image_with_histogram(f'{class_name}')
                
            else:
                ImagePlotter(output).plot_image(f'{class_name}')
        
        return output
    
class HOG_Tilation:
    def __init__(self, ref_img, chi_threshold, cell_size = (10, 10), block_size=(2,2), orientations=9):
        self.ref_img = ref_img
        self.chi_threshold = chi_threshold
        self.cell_size = cell_size
        self.block_size = block_size
        self.orientations = orientations
        self.class_name = self.__class__.__name__
    
    def apply(self, target_img):
        
        ref_img = np.squeeze(self.ref_img)
        target_img = np.squeeze(target_img)
        
        ref_fds = hog(ref_img, orientations=self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size, block_norm='L2-Hys')
        target_fds = hog(target_img, orientations=self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size, block_norm='L2-Hys')
        
        # Reshape the feature descriptor to have the same shape as the input image
        num_blocks_vertical = (self.ref_img.shape[0] - self.block_size[0] * self.cell_size[0]) // self.cell_size[0] + 1
        num_blocks_horizontal = (self.ref_img.shape[1] - self.block_size[1] * self.cell_size[1]) // self.cell_size[1] + 1
        
        ref_img = np.resize(ref_fds, (num_blocks_vertical, num_blocks_horizontal))
        
        # print(ref_img.shape)
        
        # Get the necessary padding for the feature convolution
        target_img_hpadding = ref_img.shape[0] // 2
        target_img_vpadding = ref_img.shape[1] // 2
        
        # Take the histogram of the features
        ref_hist, _ = np.histogram(ref_img.ravel(), bins=self.orientations)
        
        # Reshape the feature descriptor to have the same shape as the input image
        num_blocks_vertical = (target_img.shape[0] - self.block_size[0] * self.cell_size[0]) // self.cell_size[0] + 1
        num_blocks_horizontal = (target_img.shape[1] - self.block_size[1] * self.cell_size[1]) // self.cell_size[1] + 1
        
        # Take the feature descriptor as the new target image
        target_img = np.resize(target_fds, (num_blocks_vertical, num_blocks_horizontal))
        
        # print(target_img.shape)
        
        img = np.zeros_like(target_img)
        # nested for-loop version of the code
        for row in range(target_img_vpadding, target_img.shape[0] - target_img_vpadding):
            for col in range(target_img_hpadding, target_img.shape[1] - target_img_hpadding):
                output = self.chi_threshold_detection(target_img[row-target_img_vpadding:row+target_img_vpadding+1,
                                                    col-target_img_hpadding:col+target_img_hpadding+1], ref_hist, self.chi_threshold, self.orientations)
                
                img[row, col] = output
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img, self.class_name
    
    def chi_threshold_detection(self, target_img, ref_hist, chi_threshold, orientations):
        
        hist_x = ref_hist
        
        # Take the histogram of the features
        hist_y, _ = np.histogram(target_img.ravel(), bins=orientations)
        
        num = (hist_x - hist_y) ** 2
        denom = hist_x + hist_y
        diff = 0.5 * np.sum(num / denom)
        
        if diff < chi_threshold:
            return 255
        else:
            return 0
        
class HOG_Parallel:
    def __init__(self, ref_img, cell_size = (10, 10), block_size=(2,2), orientations=9):
        self.ref_img = ref_img
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = orientations
        self.class_name = self.__class__.__name__
    
    def apply(self, target_img):
        
        ref_img = np.squeeze(self.ref_img)
        target_img = np.squeeze(target_img)
        
        # Calculate the new dimensions based on cell size and block stride
        new_height = (target_img.shape[0] // self.cell_size[0]) * self.cell_size[0]
        new_width = (target_img.shape[1] // self.cell_size[1]) * self.cell_size[1]
        
        # Resize the image to match the HOG operation shape
        target_img = cv2.resize(target_img, (new_width, new_height))
        
        # Calculate the new dimensions based on cell size and block stride
        new_height = (ref_img.shape[0] // self.cell_size[0]) * self.cell_size[0]
        new_width = (ref_img.shape[1] // self.cell_size[1]) * self.cell_size[1]
        
        ref_img = cv2.resize(ref_img, (new_width, new_height))
        
        # Compute HOG features for the reference image
        ref_features = hog(ref_img, orientations=self.bins, pixels_per_cell=self.cell_size, cells_per_block=self.block_size, block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
        
        # Take the histogram of the features
        
        # Take the histogram of the features
        ref_hist, _ = np.histogram(ref_features.ravel(), bins=self.bins)
        # Normalize the histogram
        ref_hist = ref_hist // np.max(ref_hist)

        
        img = np.zeros_like(target_img)
        required_width = self.ref_img.shape[1] + 1
        required_height = self.ref_img.shape[0] + 1
        
        def chi_square_distance(hist1, hist2):
            return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
        
        def process_subimage(row, col):
            sub_image = target_img[row:row + required_height + 1, col:col + required_width + 1]

            target_fds = hog(sub_image, orientations=self.bins, pixels_per_cell=self.cell_size, cells_per_block=self.block_size, block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
            target_hist, _ = np.histogram(target_fds, bins=self.bins)
            target_hist = target_hist // np.max(target_hist)

            distance = chi_square_distance(target_hist, ref_hist)
            return distance, row, col


        min_distance = 1000
        best_location = (1, 1)

        with ThreadPoolExecutor() as executor:
            futures = []
            for row in range(target_img.shape[0] - required_height + 1):
                for col in range(target_img.shape[1] - required_width + 1):
                    futures.append(executor.submit(process_subimage, row, col))

            for future in futures:
                distance, row, col = future.result()
                if distance < min_distance:
                    min_distance = distance
                    best_location = (row, col)

        img = cv2.rectangle(target_img, best_location, (best_location[0] + required_height, best_location[1] + required_width), 255, 1)

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img, self.class_name

class HOG:
    def __init__(self, ref_img, cell_size = (10, 10), block_size=(2,2), orientations=9):
        self.ref_img = ref_img
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = orientations
        self.class_name = self.__class__.__name__
    
    def apply(self, target_img):
        
        ref_img = np.squeeze(self.ref_img)
        target_img = np.squeeze(target_img)
        
        # Calculate the new dimensions based on cell size and block stride
        new_height = (target_img.shape[0] // self.cell_size[0]) * self.cell_size[0]
        new_width = (target_img.shape[1] // self.cell_size[1]) * self.cell_size[1]
        
        # Resize the image to match the HOG operation shape
        ref_img = cv2.resize(self.ref_img, (new_width, new_height))
        
        # Create HOG descriptor with the specified parameters
        # The target HOG is the same shape as the ref_hog_descriptor for each pixel.
        ref_hog_descriptor = cv2.HOGDescriptor(
            _winSize=(new_width, new_height),
            _blockSize=(self.block_size[1] * self.cell_size[1], self.block_size[0] * self.cell_size[0]),
            _blockStride=(self.cell_size[1], self.cell_size[0]),
            _cellSize=(self.cell_size[1], self.cell_size[0]),
            _nbins=self.bins
        )
        
        ref_features = ref_hog_descriptor.compute(ref_img)
        
        # Take the histogram of the features
        ref_hist, _ = np.histogram(ref_features.ravel(), bins=self.bins)
        # Normalize the histogram
        ref_hist = ref_hist // np.max(ref_hist)

        
        img = np.zeros_like(target_img)
        required_width = self.ref_img.shape[1] + 1
        required_height = self.ref_img.shape[0] + 1
        
        def chi_square_distance(hist1, hist2):
            return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
        
        min_distance = 1000
        best_location = (1,1)
        
        # nested for-loop version of the code
        for row in range(target_img.shape[0] - required_height + 1):
            for col in range(target_img.shape[1] - required_width + 1):
                sub_image = target_img[row:row + required_height+1, col:col + required_width+1]
                
                target_fds = ref_hog_descriptor.compute(sub_image)
                # Take the histogram of the features
                target_hist, _ = np.histogram(target_fds, bins=self.bins)
                target_hist = target_hist // np.max(target_hist)
                
                distance = chi_square_distance(target_hist, ref_hist)
                
                output = 0
                if distance < min_distance:
                    min_distance = distance
                    best_location = (row, col)
                        
                img[row, col] = output
                
        cv2.rectangle(target_img, best_location, (best_location[0] + required_height, best_location[1] + required_width), 255, 1)
                
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img, self.class_name
    
    # @jit(nopython=True, parallel=True) # Add the jit decorator with nopython mode
    # def histogram(data, bins=10, range=None):
        
    #     data = np.array(data)

    #     # Check if the range is specified
    #     if range is None:
    #         # Use the minimum and maximum values of the data as the range
    #         range = (data.min(), data.max())

    #     # Calculate the bin edges without using np.linspace
    #     bin_edges = np.empty(bins + 1, dtype=np.float64) # Create an empty array of the desired size
    #     step = (range[1] - range[0]) / bins # Calculate the step size
    #     for i in prange(bins + 1): # Loop over the array indices
    #         bin_edges[i] = range[0] + i * step # Fill the array with linearly spaced values

    #     # Count the number of elements in each bin
    #     bin_counts = np.zeros(bins, dtype=int)
    #     for i in prange(bins):
    #         # Find the elements that fall within the current bin
    #         mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
    #         # Count the number of elements in the current bin
    #         bin_counts[i] = np.sum(mask)

    #     # Return the bin counts and bin edges
    #     return bin_counts, bin_edges
        

    # @njit(parallel=True)
    # def hog(mag: np.ndarray, ang: np.ndarray, cell_size: Tuple[int, int] = (8, 8), block_size: Tuple[int, int] = (2, 2), bins: int = 9) -> np.ndarray:
        
    #     height, width = mag.shape
        
    #     for i in prange(height):
    #         for j in prange(width):
    #             ang[i, j] = np.int32(bins * ang[i, j] / (2 * np.pi))

    #     # Calculate the histogram for each cell
    #     cell_x = width // cell_size[0]
    #     cell_y = height // cell_size[1]
    #     hist = np.zeros((cell_y, cell_x, bins), dtype=np.float32)
    #     for i in prange(cell_y):
    #         for j in prange(cell_x):
    #             # Get the sub-image for the current cell
    #             sub_mag = mag[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
    #             sub_ang = ang[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
    #               # Accumulate the histogram for the current cell
    #             for m, a in zip(sub_mag.ravel(), sub_ang.ravel()):
    #                 hist[i, j, a] += m

    #     # Normalize the histogram for each block
    #     block_x = (cell_x - block_size[0]) + 1
    #     block_y = (cell_y - block_size[1]) + 1
    #     hog = np.zeros((block_y, block_x, block_size[0] * block_size[1] * bins), dtype=np.float32)
    #     for i in prange(block_y):
    #         for j in prange(block_x):
    #             # Get the sub-histogram for the current block
    #             sub_hist = hist[i:i + block_size[0], j:j + block_size[1]].ravel()
    #             # Normalize the sub-histogram using L2-norm
    #             norm = np.sqrt(np.sum(sub_hist ** 2))
    #             hog[i, j] = sub_hist / norm

    #     # Return the flattened HOG features
    #     return hog.ravel()

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
                                (max_val - min_val)) + output_min + 0.0001

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
        img = np.array(img).astype(np.int8)

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
    