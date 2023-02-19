import numpy as np
import colorspace

# Estimates pepper noise on a single layer
def pepper_noise_estimator(img):
    count = np.count_nonzero(np.logical_or(img < 3, img > 253))
    # Return density
    return count / img.size

# Estimates gaussian noise on a single layer
def estimate_gaussian_noise_variance(image):
    
    # Compute the absolute difference between adjacent pixels in the image
    diff_image = np.abs(np.diff(image))

    # Compute the variance of the differences
    variance = np.var(diff_image)

    return variance
