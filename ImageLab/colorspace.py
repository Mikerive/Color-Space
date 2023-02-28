from PIL import Image
import cv2
import numpy as np

class ColorSpace:
    def __init__(self, img):
        self.img = np.asarray(img)
    def rgb_decomposition(self):
        # Split the input image into its blue, green, and red channels
        b, g, r = self.img[:,:,0], self.img[:,:,1], self.img[:, :, 2]
        # Return the blue, green, and red channels as a tuple
        return r, g, b
    
    # Luminosity Method
    def rgb_to_grayscale(self, ratio=[0.2989, 0.5870, 0.1140]):
        
        gray_image = np.dot(self.img[..., :3], ratio)

        gray_image = np.clip(gray_image, 0, 255).astype(np.int)
        
        return gray_image

    def rgb_to_HSV(image):
        # Convert the NumPy array to a Pillow Image
        pil_image = Image.fromarray(image)

        # Convert the image to the HSV color space
        hsv_image = pil_image.convert('HSV')

        # Convert the HSV image back to a NumPy array
        hsv_array = np.array(hsv_image)

        return hsv_array


                
    
    
