import numpy as np
from .imageutils import ImageUtil, ImagePlotter
from multiprocessing import Pool
from functools import partial

class Convolution:
    def __init__(self, img=np.full((1, 1), 1), img_name='default', hist=False):
         
        self.img = np.asarray(img)
            
        self.img_name = img_name
        
        self.hist = hist
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
            
    default_matrix = np.full((3, 3), 1)
    def kernel_operation(self, pixel, img, kernel_matrix, kernel_radius, func, *args):
        
        row, col, channel = pixel
        
        # Extract the sub_image centered at the specified pixel
        sub_image = img[row-kernel_radius:row + kernel_radius+1, col-kernel_radius:col+kernel_radius+1, channel]

        # Apply the parameter function to the sub_image
        return func(sub_image, kernel_matrix, *args)

            
    def sliding_kernel(self, kernel_matrix, func, *args, num_processes=4):
        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)
        
        img = np.array(np.copy(self.img))
        
        k_matrix = np.array(kernel_matrix)
        
        kernel_size = k_matrix.shape[0]
        
        padding_size = kernel_size // 2
        
        # Get the coordinates of the center pixel in the kernel matrix
        radius = np.array(k_matrix.shape[0]) // 2
        
        # Pad the image with zeros
        padded_image = np.pad(img, ((padding_size, padding_size),
                              (padding_size, padding_size), (0, 0)), mode='constant')

        x_coords, y_coords, z_coords = np.meshgrid(
            range(padding_size, img.shape[1]+padding_size), range(padding_size, img.shape[0]+padding_size), range(img.shape[2]), indexing='ij')

        pixel_coords = np.column_stack((np.array(y_coords).ravel(), np.array(
            x_coords).ravel(), np.array(z_coords).ravel()))
        
        with Pool(num_processes) as p:
            results = p.map(partial(self.kernel_operation, kernel_radius = radius, img=padded_image, kernel_matrix=k_matrix, func=func, *args), pixel_coords)
        
        print(np.array(results).shape)
        # Convert the results iterator to an array and reshape it to the desired shape
        output_image = np.fromiter(results, dtype=np.float32).reshape(self.img.shape)
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

        if self.hist == True:
            ImagePlotter(output_image).plot_image_with_histogram(
                title=f'{self.img_name}_n={kernel_size}')
        else:
            ImagePlotter(output_image).plot_image(
                title=f'{self.img_name}_n={kernel_size}')

        if output_image.shape[2] == 1:
            output_image = np.squeeze(output_image)
            path = ImageUtil(output_image).save_image_to_folder(
                'Image/Convolution/', f"{self.img_name}.png")
            return output_image, path
        else:
            path = ImageUtil(output_image).save_image_to_folder(
                'Image/Convolution/', f"{self.img_name}.png")
            return output_image, path
        
        

    def gaussian_kernel(n, sigma):
        x, y = np.meshgrid(np.arange(-n // 2 + 1, n // 2 + 1), np.arange(-n // 2 + 1, n // 2 + 1))
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g // g.sum()

    def weighted_arithmetic_mean(self, arr, weight_matrix=default_matrix):
        # Normalize the weight matrix
        weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)
        
        return np.dot(np.array(arr).ravel(), np.array(weight_matrix).ravel())

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
