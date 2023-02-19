import matplotlib.pyplot as plt
import numpy as np

class ColorSpace:
    def __init__(self, img):
        self.img = img
    def rgb_decomposition(self):
        # Split the input image into its blue, green, and red channels
        b, g, r = self.img[:,:,0], self.img[:,:,1], self.img[:, :, 2]
        
        # Return the blue, green, and red channels as a tuple
        return r, g, b
    
    def rgb_to_grayscale(self, ratio=[0.2989, 0.5870, 0.1140]):
        # Load the image and convert it to grayscale
        image = plt.imread(self.img)
        gray_image = np.dot(image[..., :3], ratio)
    

        # # Split the RGB image into its red, green, and blue channels
        # r, g, b = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]

        # # Merge the noisy channels back into an RGB image
        # noisy_image = np.stack([r, g, b], axis=2)
        
        return gray_image

    def rgb_to_HSV(self):
        # Convert the RGB image to HSV color space
        R, G, B = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]
        V = np.max(self.img, axis=2)
        S = (V - np.min(self.img, axis=2)) / V
        H = np.zeros_like(V)
        H[np.where(V == R)] = 60 * ((G[np.where(V == R)] - B[np.where(V == R)]) /
                                    (V[np.where(V == R)] - np.min(self.img, axis=2)[np.where(V == R)])) % 360
        H[np.where(V == G)] = 60 * (2 + (B[np.where(V == G)] - R[np.where(V == G)]) /
                                    (V[np.where(V == G)] - np.min(self.img, axis=2)[np.where(V == G)])) % 360
        H[np.where(V == B)] = 60 * (4 + (R[np.where(V == B)] - G[np.where(V == B)]) /
                                    (V[np.where(V == B)] - np.min(self.img, axis=2)[np.where(V == B)])) % 360
        return H, S, V

    
    

        
    
