import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .imageutils import *
from .colorspace import ColorSpace

class Segment:
    def __init__(self, img=np.full((10, 10), 1), name='default', hist=False):
        self.img = np.asarray(img).astype(int)
        self.img_name = name
        self.hist = hist
        
        # If 2d, make 3d
        if (self.img.ndim == 2):
            layers = [0]
            self.img = np.expand_dims(self.img, axis=2)
            
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
            end = minima[i+1] if i < len(minima) - 1 else 255
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
            img1, img2 = np.zeros_like(image), np.zeros_like(image)
            img1, img2 = image[image<threshold], image[image>threshold]
            thresholdnext = (np.mean(img1)+np.mean(img2))//2
            if abs(thresholdnext-threshold) < deltaT:
                done = True
            threshold = thresholdnext
        
        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(title=f'{self.img_name}')
        
        # Create a binary mask by comparing the image with the threshold value
        mask1, mask2 = np.zeros_like(image), np.zeros_like(image)
        
        mask1[image < threshold] = 255
        mask2[image >= threshold] = 255
        
        ImagePlotter(mask1).plot_image_with_histogram(
            title=f'mask 0:{threshold}')
        ImagePlotter(mask2).plot_image_with_histogram(
            title=f'mask {threshold}:255')
        # Clip the output image to ensure that pixel values are within [0, 255]
        output = np.clip(mask1, 0, 255).astype(np.uint8)
        output = np.clip(mask2, 0, 255).astype(np.uint8)

        return mask1, mask2
    
    
    def get_window(self, img, row, col, window_size):
        # When iterating through an image, we need to know the windows.
        height, width, depth = img.shape
        
        row_min = max(0, row - window_size // 2)
        row_max = min(height, row + window_size // 2 + 1)
        col_min = max(0, col - window_size // 2)
        col_max = min(width, col + window_size // 2 + 1)
        
        window = {
            'img_window': img[row_min:row_max, col_min:col_max],
            }
        return window
    
    def Niblack(self, window, k = -0.2):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window['img_window'], axis=(0, 1))
        stds = np.std(window['img_window'], axis=(0, 1))
        
        thresholds = means + k * stds
        
        return thresholds
    
    def Sauvola(self, window, k = 0.34, R=128):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window['img_window'], axis=(0, 1))
        stds = np.std(window['img_window'], axis=(0, 1))

        # Compute the local threshold for each channel
        thresholds = means * (1.0 + k * (-1 + stds / R))
        
        return thresholds
    
    def Bernsen(self, window):
        # Compute the mean and standard deviation for each channel separately
        maxs = np.max(window['img_window'], axis=(0, 1))
        mins = np.min(window['img_window'], axis=(0, 1))
        
        thresholds = (maxs + mins)/2
        
        return thresholds
    
    
    def Pixel_Filter(self, window_size, function, *args):
        
        # When iterating through an image, we need to know the windows.
        rows, cols, layers = self.img.shape
        # Get an array of windows around each pixel
        windows = np.array([self.get_window(self.img, row, col, window_size) for row, col in np.nditer([rows, cols])])
        
        thresholds = np.array([])
        
        for window in windows:
            thresholds = function(window, *args)
        
        mask = np.zeros_like(self.img)
        for layer in range(layers):
            mask_layer = np.zeros_like(self.img[:, :, layer])
            mask_layer[self.img[:, :, layer] > thresholds[layer::layers]] = 255
            mask[:, :, layer] = mask_layer
            
        if self.hist == True:
            ImagePlotter(mask).plot_image_with_histogram(
                title=f'{self.img_name}_{window_size}')
        else:
            ImagePlotter(mask).plot_image(
                title=f'{self.img_name}')

        return mask
    
    def global_multiple_threshold(self, minima):

        # Find the ranges of pixel values corresponding to halfway between each peak
        ranges = []
        for i in range(len(minima)):
            if i == 0:
                start = 0
            else:
                start = minima[i]

            # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
            end = minima[i+1] if i < len(minima) - 1 else 255
            ranges.append((start, end))

        # Create a binary mask for each range of pixel values
        masks = []
        for r in ranges:
            # Values within the range arr assigned 255, the rest 0
            mask = np.zeros_like(self.img)
            mask[(self.img >= r[0]) & (self.img <= r[1])] = 255
            masks.append(mask)

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image_with_histogram(
                title=f'mask {ranges[i][0]}:{ranges[i][1]}')

        return masks

    def adaptive_threshold_segmentation(self, n=30, background_difference=5, deltaT = 3):
        
        image = self.img.astype(int)
        
        image_dict = Tilation(image).split_image_nxn_sections(n)
        for i, image in enumerate(image_dict['section_list']):
            for layer in range(image.shape[2]):
                
                img_layer = image[:,:,layer]
                
                # Don't Segment if background
                if np.max(image) - np.min(image) < background_difference:
                    image_dict['section_list'][i][:, :, layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = (np.max(image)+np.min(image))//2
                    # If T is within three pixels of the last T, update and terminate while loop
                    done = False
                    while done == False:
                        img1, img2 = np.zeros_like(image), np.zeros_like(image)
                        img1, img2 = image[image < threshold], image[image > threshold]
                        thresholdnext = (np.mean(img1)+np.mean(img2))//2
                        if abs(thresholdnext-threshold) < deltaT:
                            done = True
                        threshold = thresholdnext

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255
                    image_dict['section_list'][i][:,:,layer] = np.clip(mask, 0, 255).astype(np.uint8)
                
        Tilation(name=f'adaptive_seg_{self.img_name} {n}:{background_difference}').merge_sections_into_image(image_dict)
        return image
    
    def Niblack_Local_Thresholding_segmentation(self, n=30, k=-0.2, background_difference=0):
        image = np.copy(self.img.astype(int))
        # Show the original image and its histogram

        image_dict = Tilation(image).split_image_nxn_sections(n)
        
        for i, image in enumerate(image_dict['section_list']):
            for layer in range(image.shape[2]):
                
                img_layer = image[:, :, layer]
                
                # Make Uniform if background
                if np.max(img_layer) - np.min(img_layer) < background_difference:
                    image_dict['section_list'][i][:, :,
                                                  layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = np.mean(img_layer) + k * np.std(img_layer)
                    
                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255
                    
                    #imagedict -> index of image -> layer of image
                    image_dict['section_list'][i][:,:,layer] = np.clip(mask, 0, 255).astype(np.uint8)
                
        Tilation(name=f'Niblack_seg_{self.img_name} {n}:{k}', hist=False).merge_sections_into_image(image_dict)
        return image
    
    def Sauvola_Local_Thresholding_segmentation(self, n=30, k=0.34, R = 128, background_difference = 0):
        image = np.copy(self.img.astype(int))

        image_dict = Tilation(image).split_image_nxn_sections(n)

        for i, image in enumerate(image_dict['section_list']):
            for layer in range(image.shape[2]):
                
                img_layer = image[:, :, layer]
                
                # Make Uniform if background
                if np.max(img_layer) - np.min(img_layer) < background_difference:
                    image_dict['section_list'][i][:, :, layer] = np.full(img_layer.shape, 255)
                else: 
                    threshold = np.mean(img_layer)*(1.0 + k*(-1 + np.std(img_layer)/R))

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255

                    #imagedict -> index of image -> layer of image
                    image_dict['section_list'][i][:, :, layer] = np.clip(
                        mask, 0, 255).astype(np.uint8)

        Tilation(name=f'Sauvola_seg_{self.img_name} {n}:{k}', hist=False).merge_sections_into_image(
            image_dict)
        return image
    
    def Bernsen_Local_Thresholding_segmentation(self, n=30, background_difference=0):
        image = np.copy(self.img.astype(int))

        image_dict = Tilation(image).split_image_nxn_sections(n)

        for i, image in enumerate(image_dict['section_list']):
            for layer in range(image.shape[2]):

                img_layer = image[:, :, layer]
                
                # Make Uniform if background
                if np.max(img_layer) - np.min(img_layer) < background_difference:
                    image_dict['section_list'][i][:, :, layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = (np.max(img_layer) + np.min(img_layer)) * 0.5

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255

                    #imagedict -> index of image -> layer of image
                    image_dict['section_list'][i][:, :, layer] = np.clip(
                        mask, 0, 255).astype(np.uint8)

        Tilation(name=f'Bernsen_seg_{self.img_name} {n}', hist = False).merge_sections_into_image(
            image_dict)
        return image
