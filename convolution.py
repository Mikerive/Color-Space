import numpy as np

def convolve(image, weight_matrix, convfunction):
    # Get the dimensions of the image and the weight matrix
    image_height, image_width, image_channels = image.shape
    weight_height, weight_width = weight_matrix.shape

    # Pad the image with zeros
    padding_size = int((weight_height - 1) / 2)
    padded_image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Iterate over each pixel in the image
    for i in range(padding_size, image_height + padding_size):
        for j in range(padding_size, image_width + padding_size):
            # Extract the neighborhood around the current pixel
            neighborhood = padded_image[i - padding_size:i + padding_size + 1, j - padding_size:j + padding_size + 1, :]

            # Apply the weight matrix to the neighborhood
            convolved_pixel = convfunction(neighborhood, weight_matrix)

            # Set the output pixel to the convolved value
            output_image[i - padding_size, j - padding_size, :] = convolved_pixel

    return output_image


def weighted_arithmetic_mean(arr, weight_matrix = [[1, 1, 1],[1, 1, 1],[1, 1, 1]]):
    
    # Normalize the weight matrix
    weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)

    # Compute the weighted sum of the neighborhood values
    weighted_sum = np.sum(arr * weight_matrix)
    
    return weighted_sum


def weighted_geometric_mean(arr, weight_matrix=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
    # Normalize the weight matrix
    weight_matrix = np.array(weight_matrix) / np.sum(weight_matrix)

    # Compute the weighted product of the neighborhood values
    weighted_product = np.prod(arr ** weight_matrix)

    # Compute the weighted geometric mean
    weighted_geometric_mean = np.power(weighted_product, 1.0 / np.sum(weight_matrix))

    return weighted_geometric_mean


def weighted_median(arr, weight_matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
    
    intermediate_array = np.array([])
    
    # Iterate over each pixel in the intermediate array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(weight_matrix[i][j]):
                intermediate_array = np.append[arr[i][j]]
            
    return np.median(intermediate_array)
