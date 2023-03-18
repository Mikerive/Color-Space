import cv2
import numpy as np
from PIL import Image

from .imageutils import ImagePlotter
from .ImageProcessing import Tilation


class Segment:
    def __init__(self, image_path, name='default', hist=False):
        image = Image.open(image_path).convert('L')
        
        self.img = np.array(image).astype(np.int32)
        self.img_name = name
        self.hist = hist

        # If 2d, make 3d
        if (self.img.ndim == 2):
            self.img = np.expand_dims(self.img, axis=2)

    def adaptive_multiple_threshold(self, distance=10, width=8):
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
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image_with_histogram(
                title=f'mask {ranges[i][0]}:{ranges[i][1]}')

        return masks

    def apply_threshold(self, threshold_value):

        # Apply thresholding
        output = (self.img > threshold_value).astype(np.uint8) * 255

        # Display the output image
        ImagePlotter(output).plot_image(
            title=f'Threshold value: {threshold_value}')

        return output

    def adaptive_global_threshold(self, mode='mean', deltaT=3):
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
            _, output = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return output

        # If T is within three pixels of the last T, update and terminate while loop
        done = False
        while done == False:
            img1, img2 = np.zeros_like(image), np.zeros_like(image)
            img1, img2 = image[image < threshold], image[image > threshold]
            thresholdnext = (np.mean(img1)+np.mean(img2))//2
            if abs(thresholdnext-threshold) < deltaT:
                done = True
            threshold = thresholdnext

        # Show the original image and its histogram
        ImagePlotter(self.img).plot_image_with_histogram(
            title=f'{self.img_name}')

        # Create a binary mask by comparing the image with the threshold value
        mask1, mask2 = np.zeros_like(image), np.zeros_like(image)

        mask1[image < threshold] = 255
        mask2[image >= threshold] = 255

        ImagePlotter(mask1).plot_image_with_histogram(
            title=f'0:{threshold}')
        ImagePlotter(mask2).plot_image_with_histogram(
            title=f'{threshold}:255')
        # Clip the output image to ensure that pixel values are within [0, 255]
        output = np.clip(mask1, 0, 255).astype(np.uint8)
        output = np.clip(mask2, 0, 255).astype(np.uint8)

        return mask1, mask2

    # def get_window(self, img, row, col, window_size):
    #     # When iterating through an image, we need to know the windows.
    #     height, width, depth = img.shape

    #     row_min = max(0, row - window_size // 2)
    #     row_max = min(height, row + window_size // 2 + 1)
    #     col_min = max(0, col - window_size // 2)
    #     col_max = min(width, col + window_size // 2 + 1)

        return img[row_min:row_max, col_min:col_max, :]

    def Pixel_Filter(self, window_size, func_type='Sauvola', **kwargs):
        # Takes Niblack, Sauvola, and Bernsen filter functions

        img = np.copy(self.img)
        # Calculate the padding size based on the window size
        padding_size = int(window_size.shape[0] // 2)

        # Pad the image with zeros
        padded_image = np.pad(img, ((padding_size, padding_size),
                              (padding_size, padding_size), (0, 0)), mode='constant')

        print(padded_image.shape)

        # filter_func = partial(func,
        #         k=kwargs.get('k'),
        #         R=kwargs.get('R')
        #         )

        if func_type == 'Niblack':
            func = Niblack
        elif func_type == 'Sauvola':
            func = Sauvola
        else:
            func = Bernsen

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

        @jit(nopython=True)
        def Bernsen(window, row, col, channel):
            # Compute the mean and standard deviation for each channel separately
            maxs = np.max(window['img_window'])
            mins = np.min(window['img_window'])

            thresholds = (maxs + mins)/2
            if window[row, col, channel] > thresholds:
                return 255
            else:
                return 0

        output = np.zeros_like(img)
        output = [func(padded_image[row-padding_size:row+padding_size+1, col-padding_size:col+padding_size+1, channel], row, col, channel)
                  for row in range(padding_size, padded_image.shape[0] - padding_size)
                  for col in range(padding_size, padded_image.shape[1] - padding_size)
                  for channel in range(0, img.shape[2])]

        output = np.clip(output, 0, 255).astype(np.uint8)

        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(
                title=f'{self.img_name}_{window_size}')
        else:
            ImagePlotter(output).plot_image(
                title=f'{self.img_name}')

        return output

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
        ImagePlotter(self.img).plot_image(
            title=f'{self.img_name}')

        # Show the identified objects
        for i in range(len(masks)):
            ImagePlotter(masks[i]).plot_image(
                title=f'Map {ranges[i][0]}:{ranges[i][1]}')

        return masks

    def adaptive_threshold_segmentation(self, n=30, background_difference=5, deltaT=3):

        image = self.img.astype(int)

        sections, image_dict = Tilation(image).split_image_nxn_sections(n)
        for i, image in enumerate(sections):
            for layer in range(image.shape[2]):

                img_layer = image[:, :, layer]

                # Don't Segment if background
                if np.max(image) - np.min(image) < background_difference:
                    sections[i][:, :,
                                                  layer] = np.full(img_layer.shape, 255)
                else:
                    threshold = (np.max(image)+np.min(image))//2
                    # If T is within three pixels of the last T, update and terminate while loop
                    done = False
                    while done == False:
                        img1, img2 = np.zeros_like(image), np.zeros_like(image)
                        img1, img2 = image[image <
                                           threshold], image[image > threshold]
                        thresholdnext = (np.mean(img1)+np.mean(img2))//2
                        if abs(thresholdnext-threshold) < deltaT:
                            done = True
                        threshold = thresholdnext

                    # Create a binary mask by comparing the image with the threshold value
                    mask = np.zeros_like(img_layer)

                    mask[img_layer >= threshold] = 255
                    sections[i][:, :, layer] = np.clip(
                        mask, 0, 255).astype(np.uint8)

        Tilation(name=f'adaptive_seg_{self.img_name} {n}:{background_difference}').merge_sections_into_image(
            image_dict)
        return image
