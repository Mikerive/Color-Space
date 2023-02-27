import cv2
import numpy as np
import matplotlib as plt
from skimage.transform import resize

__all__ = ['']

class Filters:
    def __init__(self, img = None) -> None:
        pass
    def reduce_size(img, factor=2):
        # Resize the image to half its original size
        height, width = img.shape[:factor]
        img_resized = resize(img, (height//factor, width //
                            factor), anti_aliasing=False)
        return img_resized

    def contrast_stretch(img):
        # Convert the image to a numpy array
        img = np.array(img)

        # Calculate the minimum and maximum values of the image
        min_val = np.min(img)
        max_val = np.max(img)

        # New minimum and maximum values
        output_min = 0
        output_max = 255

        # Contrast Stretch = Current Delta * Stretching factor + new_min
        img = (img - min_val) * ((output_max - output_min) /
                                (max_val - min_val)) + output_min

        # Convert the image back to 8-bit unsigned integer format
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def gamma_correction(img, gamma=1.0, alpha=0.0):
        # Convert the image to a numpy array
        img = np.array(img)

        # Normalize the image to the range [0, 1]
        img = img / 255.0

        # Apply the gamma correction
        img = np.power(img+alpha, gamma)

        # Scale the image back to the range [0, 255]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

    def histogram_equalization(img):
        # Convert the image to a grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the cdf
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()

        # Calculate the normalized cdf
        cdf_normalized = cdf / cdf.max()

        # Map the pixel values to the normalized cumulative distribution
        img_equalized = np.interp(gray, bins[:-1], cdf_normalized)

        # Convert the image back to 8-bit unsigned integer format
        img_equalized = np.clip(img_equalized*255, 0, 255).astype(np.uint8)

        # Convert the image back to BGR format
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)

        return img_equalized


