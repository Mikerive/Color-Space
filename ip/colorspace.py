import cv2
import numpy as np
import colorconversionfunctions as ccf

class ColorSpace:
    def __init__(self, img):
        self.img = np.array(img).astype(np.uint8)
    def rgb_decomposition(self):
        # Split the input image into its blue, green, and red channels
        b, g, r = self.img[:,:,0], self.img[:,:,1], self.img[:, :, 2]
        
        # Return the blue, green, and red channels as a tuple
        return r, g, b
    
    # Luminosity Method
    def rgb_to_grayscale(self, ratio=[0.2989, 0.5870, 0.1140]):
        
        gray_image = np.dot(self.img[..., :3], ratio)
    

        # # Split the RGB image into its red, green, and blue channels
        # r, g, b = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]

        # # Merge the noisy channels back into an RGB image
        # noisy_image = np.stack([r, g, b], axis=2)
        
        # Add the noise to the image and cast to int
        gray_image = np.clip(gray_image, 0, 255).astype(np.int)
        
        return gray_image

    # def rgb_to_HSV(self):
    #     img_height, img_width, img_depth = self.img.shape
        
    #     output = np.ones(self.img.shape)
        
    #     for i in range(0, img_height):
    #         for j in range(0, img_width):
    #             h, s, v = ccf.rgb_to_hsv(self.img[i, j, 0], self.img[i, j, 1], self.img[i, j, 2])
    #             output[i,j,0] = h
    #             output[i,j,1] = s
    #             output[i,j,2] = v
    #     return output


    # def rgb_to_HSV(self):
    #     output = np.ones(self.img.shape)
    #     args = (self.img[..., 0], self.img[..., 1], self.img[..., 2])
    #     output[..., 0], output[..., 1], output[..., 2] = np.apply_along_axis(ccf.rgb_to_hsv, 2, *args)
    #     return output
    
    def rgb_to_HSV(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

                
    
    
