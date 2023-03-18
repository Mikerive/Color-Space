import cv2
import numpy as np
from PIL import Image
from numba import jit


from .imageutils import ImagePlotter, ImageUtil
from .ImageProcessing import Tilation

__all__ = ['Segment', 'Adaptive_Multiple_Threshold', 'Adaptive_Multiple_Threshold',
           'Adaptive_Global', 'Pixel_Filter', 'Global_Multiple_Threshold', 'Global_Threshold',
           'Tiled_Adaptive_Threshold_Segmentation']

class Segment:
    def __init__(self, image_path, folder_name, hist=False):
        self.image_path = image_path
        self.folder_name = folder_name
        self.hist = hist

    def process(self, operator):
        # Open the input image
        image = Image.open(self.image_path)
        image_array = np.array(image)

        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=2)

        # Apply the operator to the input image
        output, class_name = operator.apply(image_array)

        path = ImageUtil(output).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{class_name}.png")

        return output, path

class Adaptive_Multiple_Threshold:
    def __init__(self, distance=10, width=8):
        self.class_name = self.__class__.__name__
        self.distance = distance
        self.width = width
    def apply(self):
        # Calculate the histogram of the image
        hist, bins = np.histogram(self.img.flatten(), bins=256, range=(0, 255))

        # Find the peaks in the histogram using the find_peaks function from SciPy
        from scipy.signal import find_peaks

        # Find peaks with a minimum height of 1/4 of the maximum value
        minima, _ = find_peaks(-hist, distance=self.distance, width=self.width)

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

        return masks, self.class_name

class Global_Threshold:
    def __init__(self, threshold_value):
        self.class_name = self.__class__.__name__
        self.threshold_value = threshold_value

    def apply(self):

        # Apply thresholding
        output = (self.img > self.threshold_value).astype(np.uint8) * 255

        # Display the output image
        ImagePlotter(output).plot_image(
            title=f'Threshold value: {self.threshold_value}')

        return output, self.class_name

class Adaptive_Global_Threshold:
    """
    Applies global thresholding to an input grayscale image.
    
    :param image: A grayscale image as a NumPy array.
    :param threshold: The threshold value.
    :return: The thresholded image as a binary NumPy array.
    """
    def __init__(self, mode='mean', deltaT=3):
        self.class_name = self.__class__.__name__
        self.deltaT = deltaT
        self.mode = mode
    
    def apply(self, img):
        image = img.astype(int)
        threshold = None

        if self.mode == 'mean':
            threshold = (np.max(image)+np.min(image))//2
        elif self.mode == 'median':
            threshold = np.median(image)
        # Minimizes the intra-class variance of the two resulting classes (foreground and background)
        elif self.mode == 'otsu':
            _, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return output

        # If T is within three pixels of the last T, update and terminate while loop
        done = False
        while done == False:
            img1, img2 = np.zeros_like(image), np.zeros_like(image)
            img1, img2 = image[image < threshold], image[image > threshold]
            thresholdnext = (np.mean(img1)+np.mean(img2))//2
            if abs(thresholdnext-threshold) < self.deltaT:
                done = True
            threshold = thresholdnext

        # Show the original image and its histogram
        ImagePlotter(img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Create a binary mask by comparing the image with the threshold value
        mask2 = np.zeros_like(image)

        mask2[image > threshold] = 255

        ImagePlotter(mask2).plot_image_with_histogram(
            title=f'{threshold}:255')
        # Clip the output image to ensure that pixel values are within [0, 255]
        output = np.clip(mask2, 0, 255).astype(np.uint8)

        return mask2, self.class_name

class Pixel_Filter:
    def __init__(self, window_size, func_type='Sauvola', hist = False):
        self.class_name = self.__class__.__name__
        self.window_size = window_size
        self.func_type = func_type
        self.hist = hist
    
    @staticmethod
    @jit(nopython=True)
    def Niblack(window, row, col, channel, k=-0.2):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window)
        stds = np.std(window)

        thresholds = means + k * stds

        if window[row, col, channel] > thresholds:
            return 255
        else:
            return 0

    @staticmethod
    @jit(nopython=True)
    def Sauvola(window, row, col, channel, k=0.34, R=128):
        # Compute the mean and standard deviation for each channel separately
        means = np.mean(window)
        stds = np.std(window)

        # Compute the local threshold for each channel
        thresholds = means * (1.0 + k * (-1 + stds / R))

        if window[row, col, channel] > thresholds:
            return 255
        else:
            return 0

    @staticmethod
    @jit(nopython=True)
    def Bernsen(window, row, col, channel):
        # Compute the mean and standard deviation for each channel separately
        maxs = np.max(window)
        mins = np.min(window)

        thresholds = (maxs + mins)/2
        if window[row, col, channel] > thresholds:
            return 255
        else:
            return 0
        
    def apply(self, img):
        # Takes Niblack, Sauvola, and Bernsen filter functions
        # Calculate the padding size based on the window size
        padding_size = int(self.window_size.shape[0] // 2)

        # Pad the image with zeros
        padded_image = np.pad(img, ((padding_size, padding_size),
                              (padding_size, padding_size), (0, 0)), mode='constant')

        print(padded_image.shape)

        if self.func_type == 'Niblack':
            func = self.Niblack
        elif self.func_type == 'Sauvola':
            func = self.Sauvola
        else:
            func = self.Bernsen

        output = np.zeros_like(img)
        output = [func(padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel], row, col, channel)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0, img.shape[2])]

        output = np.clip(output, 0, 255).astype(np.uint8)

        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(
                title=f'{self.img_name}_{self.window_size}')
        else:
            ImagePlotter(output).plot_image(
                title=f'{self.img_name}')

        return output, self.class_name

class Global_Multiple_Threshold:
    def __init__(self, minima):
        self.class_name = self.__class__.__name__
        self.minima = minima
    def apply(self, img):
        # Find the ranges of pixel values corresponding to halfway between each peak
        ranges = []
        for i in range(len(self.minima)):
            if i == 0:
                start = 0
            else:
                start = self.minima[i]

            # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
            end = self.minima[i+1] if i < len(self.minima) - 1 else 255
            ranges.append((start, end))

        # Create a binary mask for each range of pixel values
        masks = []
        for r in ranges:
            # Values within the range arr assigned 255, the rest 0
            mask = np.zeros_like(img)
            mask[(img >= r[0]) & (img <= r[1])] = 255
            masks.append(mask)

        # Show the original image and its histogram
        ImagePlotter(img).plot_image(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image(
                title=f'Map {ranges[i][0]}:{ranges[i][1]}')

        return masks, self.class_name

class Tiled_Adaptive_Threshold_Segmentation:
    def __init__(self, n=30, background_difference=5, deltaT=3):
        self.class_name = self.__class__.__name__
        self.n = n
        self.background_difference=background_difference
        self.deltaT = deltaT

    def apply(self, image):
        sections, image_dict = Tilation(image).split_image_nxn_sections(self.n)
        for i, image in enumerate(sections):
            for layer in range(image.shape[2]):

                img_layer = image[:, :, layer]

                # Don't Segment if background
                if np.max(image) - np.min(image) < self.background_difference:
                    sections[i][:, :,layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = (np.max(image)+np.min(image))//2
                    # If T is within three pixels of the last T, update and terminate while loop
                    done = False
                    while done == False:
                        img1, img2 = np.zeros_like(image), np.zeros_like(image)
                        img1, img2 = image[image <
                                           threshold], image[image > threshold]
                        thresholdnext = (np.mean(img1)+np.mean(img2))//2
                        if abs(thresholdnext-threshold) < self.deltaT:
                            done = True
                        threshold = thresholdnext

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255
                    sections[i][:, :, layer] = np.clip(
                        mask, 0, 255).astype(np.uint8)

        Tilation(name=f'adaptive_seg_{self.img_name} {self.n}:{self.background_difference}').merge_sections_into_image(
            image_dict)
        return image, self.class_name
