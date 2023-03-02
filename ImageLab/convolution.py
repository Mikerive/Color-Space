import numpy as np
from .imageutils import ImageUtil

class Convolution:
    def __init__(self, img, img_name):
        self.img = np.asarray(img)
        self.img_name = img_name
        
    def convolve(self, weight_matrix, convfunction, layers = [0,1,2]):
        
        # If 2d, make 3d
        if (self.img.ndim == 2):
            layers = [0]
            self.img = np.expand_dims(self.img, axis=2)
            
        # Get the dimensions of the image and the weight matrix
        image_height, image_width, depth = self.img.shape
        weight_height, weight_width = weight_matrix.shape

        # Pad the image with zeros
        padding_size = int((weight_height - 1) / 2)
        padded_image = np.pad(self.img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')

        # Initialize the output image
        output_image = np.zeros_like(self.img)
        
        # For each specified layer
        for layer in layers:
            # Iterate over each pixel in the image
            for i in range(padding_size, image_height + padding_size):
                for j in range(padding_size, image_width + padding_size):
                    # Extract the neighborhood around the current pixel
                    neighborhood = padded_image[i - padding_size:i + padding_size + 1, j - padding_size:j + padding_size + 1, layer]

                    # Apply the weight matrix to the neighborhood
                    convolved_pixel = convfunction(neighborhood, weight_matrix)

                    # Set the output pixel to the convolved value
                    output_image[i - padding_size, j - padding_size, layer] = convolved_pixel
                    
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        
        if output_image.shape[2] == 1:
            output_image = np.squeeze(output_image)
            path = ImageUtil(output_image).save_image_to_folder(
                'Image/Convolution/', f"{self.img_name}.png")
            return output_image, path
        else:
            path = ImageUtil(output_image).save_image_to_folder('Image/Convolution/', f"{self.img_name}.png")
            return output_image, path


# Convenience class to import many functions with one call


class Conv_Functions:
    default_matrix = np.full((3, 3), 1)
    def __init__(self):
        pass

    def weighted_arithmetic_mean(self, arr, weight_matrix=default_matrix):
        
        # Normalize the weight matrix
        weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)

        # Compute the weighted sum of the neighborhood values
        weighted_sum = np.sum(arr * weight_matrix)
        
        return weighted_sum

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
