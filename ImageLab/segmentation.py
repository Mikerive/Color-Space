import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .imageutils import *
from .colorspace import ColorSpace

class Segment:
    def __init__(self, img=np.full((10, 10), 1), name='default'):
        self.img = np.asarray(img).astype(int)
        self.img_name = name
            
    def segment_image(self, distance=10, width=8):
        # Calculate the histogram of the image
        hist, bins = np.histogram(self.img.flatten(), bins=256, range=(0, 255))
        
        # Find the peaks in the histogram using the find_peaks function from SciPy
        from scipy.signal import find_peaks
        
        # Find peaks with a minimum height of 1/4 of the maximum value
        minima, _ = find_peaks(-hist, distance=distance, width=width)
        
        # Find the ranges of pixel values corresponding to halfway between each peak
        ranges = []
        for i in range(len(minima)):
            if i == 0:
                start = 0
            else:
                start = minima[i]
            
            # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
            end = minima[i+1] // 2 if i < len(minima) - 1 else 255
            ranges.append((start, end))

        # Create a binary mask for each range of pixel values
        masks = []
        for r in ranges:
            # Values within the range arr assigned 255, the rest 0
            mask = np.zeros_like(self.img)
            mask[(self.img >= r[0]) & (self.img <= r[1])] = 255
            masks.append(mask)

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image_with_histogram(
                title=f'mask {ranges[i][0]}:{ranges[i][1]}')
        
        return masks


    def global_threshold(self, mode = 'mean', deltaT = 3):
        """
        Applies global thresholding to an input grayscale image.
        
        :param image: A grayscale image as a NumPy array.
        :param threshold: The threshold value.
        :return: The thresholded image as a binary NumPy array.
        """
        image = self.img.astype(int)
        threshold = None
        
        if mode == 'mean':
            threshold = (np.max(image)+np.min(image))//2
        elif mode == 'median':
            threshold = np.median(image)
        # Minimizes the intra-class variance of the two resulting classes (foreground and background)
        elif mode == 'otsu':
            _, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return output
        
        # If T is within three pixels of the last T, update and terminate while loop
        done = False
        while done == False:
            img1, img2 = np.zeros_like(image)
            img1, img2 = image[image<threshold], image[image>threshold]
            thresholdnext = (np.mean(img1)+np.mean(img2))//2
            if abs(thresholdnext-threshold) < deltaT:
                done = True
            threshold = thresholdnext
        
        # Create a binary mask by comparing the image with the threshold value
        mask = np.zeros_like(image)
        mask[image >= threshold] = 1

        # Multiply the mask by 255 to obtain a binary image with 0s and 255s
        output = mask * 255
        
        # Clip the output image to ensure that pixel values are within [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output


    def global_multiple_threshold(image, thresholds, mode = 'mean'):
        
        if mode == 'mean':
            threshold = (np.max(image)+np.min(image))/2
        
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = ColorSpace(image)

        # Initialize the segmented image to all black pixels
        segmented = np.zeros_like(image)

        # For each threshold, create a binary image and merge it with the segmented image
        for threshold in thresholds:
            binary = np.where(image >= threshold, 255, 0)
            segmented = np.maximum(segmented, binary)

        # Apply a median filter to reduce noise
        segmented = cv2.medianBlur(segmented, 3)

        return segmented
