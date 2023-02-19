import numpy as np
import matplotlib.pyplot as plt


# Noise should be multiplied by the maximum value of pixels before being added to an image.

def gaussian_noise(image, mean, variance):
    # Create an array of random Gaussian noise with the specified mean and variance
    noise = np.random.normal(mean, np.sqrt(variance), image.shape)
    
    # Clip noise and cast to int
    noise = np.clip(noise, 0, 255).astype(np.int)

    return noise


def add_gaussian_noise(image, mean, variance):
    # Generate Gaussian noise
    noise = gaussian_noise(image, mean, variance)

    # Add the noise to the image and cast to int
    noisy_image = np.clip(image + noise, 0, 255).astype(np.int)

    return noisy_image


def create_salt_and_pepper_noise(image, density):
    # Create a noise array with the same shape as the image
    noise = np.random.random(image.shape[:2])
    
    noise[noise < density/2] = 0
    noise[noise > 1-density/2] = 255
    
    # Clip the values of the noisy image to the valid range of pixel values (0 to 255 for uint8 images)
    noise = np.clip(noise, 0, 255).astype(np.int)

    return noise
