import cv2
import numpy as np

class NoiseOverlay:
    def __init__(self, img):
        self.img = np.array(img)
        if np.ndim(self.img) == 2:
            self.img = np.expand_dims(self.img, axis=2)
        
    # Noise is a single layer based on an image
    def gaussian_noise(self, mean, variance):
        # Create an array of random Gaussian noise with the specified mean and variance
        noise = np.random.normal(mean, np.sqrt(variance), self.img.shape[:2])
        
        # Clip noise and cast to int
        noise = np.clip(noise, 0, 255).astype(np.int)

        return noise

    def add_gaussian_noise(self, mean, variance, channels = [0, 1, 2], type = 'image'):
        if type == 'noise':
            # Generate Gaussian noise
            noise = self.gaussian_noise(self.img, mean, variance)
            
            # Add the noise to the image and cast to int
            noise = np.clip(noise, 0, 255).astype(np.int)
            
            return noise
        
        if type == 'image':
            noisy_image = np.array(self.img)
            
            for i in channels:
                # Generate Gaussian noise
                noise_layer = self.gaussian_noise(mean, variance)
                
                noise_layer = self.img[:,:,i] + noise_layer
                
                # Add the noise to the image and cast to int
                noise_layer = np.clip(noise_layer, 0, 255).astype(np.int)
                
                noisy_image[:,:,i] = noise_layer
                
            return noisy_image
        else:
            raise ValueError("type can either be image or noise")
        
    def salt_and_pepper_noise(self, density):
        noise_mask = np.random.random(self.img.shape[:2])

        noise = np.full(self.img.shape[:2], 125)

        noise[noise_mask < density/2] = 1
        noise[noise_mask > 1-density/2] = 255

        # Clip the values of the noisy image to the valid range of pixel values (0 to 255 for uint8 images)
        noise = np.clip(noise, 0, 255).astype(np.int)
        return noise

    def add_salt_and_pepper_noise(self , density, channels = None):
        
        if channels is None:
            channels = range(self.img.shape[2])
        
        noisy_image = np.copy(self.img)
        
        # For each channel to add noise to...
        for i in channels:
            # Noise Layer
            noise_mask = np.random.random(self.img.shape)
            # Image layer
            image_layer = noisy_image[:,:,i]
            
            # Salt and Pepper
            image_layer[noise_mask[:,:,i] < density/2] = 1
            image_layer[noise_mask[:,:,i] > 1-density/2] = 255

            # Clip the values of the noisy image to the valid range of pixel values (0 to 255 for uint8 images)
            image_layer = np.clip(image_layer, 0, 255).astype(np.uint8)
            
            # Assign image layer to the correct layer of noisy image
            noisy_image[:,:,i] = image_layer
            
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
        return noisy_image

    def add_equation_noise(self, equation = '0.4x+3', direction='right', magnitude=50):
        """
        Add noise to an image, increasing or decreasing from left to right based on an equation.

        Parameters:
            img (ndarray): The input image.
            equation (str): A Python expression that takes the x-coordinate as input.
            direction (str): The direction in which the noise increases or decreases. Either 'left' or 'right'.
            magnitude (float): The maximum magnitude of the noise.

        Returns:
            ndarray: The noisy image.
        """

        # Get image size and create x-coordinate grid
        height, width = self.img.shape[:2]
        x = np.arange(width)

        # Evaluate equation for each x-coordinate
        y = eval(equation)

        # Scale y-coordinates to the range [0, 1]
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Scale y-coordinates to the range [-magnitude, +magnitude]
        y = (y - 0.5) * 2 * magnitude

        # Create noise image with random values in the range [-magnitude, +magnitude]
        noise = np.random.rand(height, width) * 2 * magnitude - magnitude

        # Apply noise to image in the specified direction
        if direction == 'left':
            noise *= -1

        noisy_img = self.img.copy()
        for i in range(height):
            noisy_img[i, :] += y[i] * noise[i, :]

        # Clip values to the range [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return noisy_img
