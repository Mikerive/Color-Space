'''
Preprocessing - Isolate power lines from the image
RGB Contrast Stretch
RGB - filter out areas of the image that have a high ratio of green, red, or blue.
Areas that have a high ratio of green vs blue are likely the ground.
The sky can either be blue or red and thus often tints the powerlines. This means that filtering out red and blue is often situational. This can change depending on the time that the image was taken.
Inversion - done regardless of the day for powerlines.
Power Lines and Power towers are darker than their surroundings in most images.
Apply gamma correction (2)
Gamma correction (2) suppresses the differences between features that have a lower value and enhances the difference between features with a higher value. This is desirable.
Grayscale - improves computational efficiency during segmentation. Once the color filtering is done, convert to grayscale.
Morphological closing operation 3x3
Used to make edges better stand out. Power lines are thin and often don’t have enough difference between the background and the power lines to stand out.
Calculate Edge Direction map using fourier edge map.

'''

import cv2
import numpy as np
import os
from skimage.util import view_as_windows

def Sauvola_Threshold(img, kernel_size, stride=1, k=0.34, R=128):
    window_shape = (kernel_size, kernel_size)
    windows = view_as_windows(img, window_shape, step=stride)

    # Calculate means and standard deviations for each window
    means = np.mean(windows, axis=(2, 3))
    stds = np.std(windows, axis=(2, 3))

    # Compute the local threshold for each channel
    thresholds = means * (1.0 + k * (-1 + stds / R))

    # Get the center pixels of each window
    padding = kernel_size // 2
    center_pixels = windows[:, :, padding, padding]

    # Create the output image by applying the threshold
    output_image = np.where(center_pixels > thresholds, 255, 0)

    return output_image.astype(np.uint8)

def sliding_window_diff(img, kernel_size, stride = 1):
    window_shape = (kernel_size, kernel_size)
    # Define the border sizes
    top_border, bottom_border, left_border, right_border = kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2
    # Perform the symmetric padding with zeros
    padded_img = cv2.copyMakeBorder(img, top=top_border, bottom=bottom_border, left=left_border, right=right_border, borderType=cv2.BORDER_CONSTANT, value=0)

    # Create sliding windows of the padded image
    windows = view_as_windows(padded_img, window_shape, step=stride)

    # Compute the max and min pixel values in each window
    window_max = np.amax(windows, axis=(2, 3))
    window_min = np.amin(windows, axis=(2, 3))

    # Compute the pixel differences and clip to the range [0, 1]
    diffs = window_max - window_min

    return diffs.astype(np.uint8)

#RGB Contrast Stretch
def rgb_contrast_stretch(img):
    '''
    Apply contrast stretch on the input RGB image
    '''
    # Split the image into 3 channels
    b, g, r = cv2.split(img)

    # Apply contrast stretch on each channel
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    # Merge the channels back into an RGB image
    result = cv2.merge((b, g, r))

    return result

# This is a better filter in sunny images. In the afternoon, the sun redens and makes the filter remove too much red pixels.
def green_and_red_filter(cv_image: np.ndarray, ratio: float, kernel_size = 12):
    # Get the green, red, and blue values of all pixels
    bgr_values = cv_image[..., :3]

    # Compute the ratio of green vs blue for all pixels
    green_blue_ratios = bgr_values[..., 1] / bgr_values[..., 0]

    # Compute the ratio of red vs blue for all pixels
    red_blue_ratios = bgr_values[..., 2] / bgr_values[..., 0]

    # Set pixels with ratio greater than ratio to white
    white_mask = (green_blue_ratios > ratio) # | (red_blue_ratios > ratio)
      # Apply a kernel to compute the mean value of the kernel area
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean_filtered_image = cv2.filter2D(cv_image, -1, kernel)

    # Replace pixels that meet the ratio criteria with the mean value of the kernel area
    cv_image[white_mask] = mean_filtered_image[white_mask]
    return cv_image

def adaptive_threshold(image):
    """
    Apply adaptive global thresholding on a grayscale image.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 2, 5)
    
    return threshold_image

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
# if a region of the image is too busy, remove that pixel - it isn't a wire
def average_threshold(img, threshold):
    kernel = np.ones((9,9), dtype=np.uint8)
    avg = cv2.filter2D(img, -1, kernel) / 81
    img[avg>threshold] = 0
    return img
# If a region of the iamge has a variance that is too high, remove that pixel
def variance_threshold(img, threshold):
    kernel = np.ones((5,5), dtype=np.uint8)
    # computing variance using filter2D and square operation
    meanSq = cv2.filter2D(img, -1, kernel)**2 // 25
    sqMean = cv2.filter2D(img**2, -1, kernel) // 25
    var = sqMean - meanSq
    # thresholding
    img[var>threshold] = 0
    return img
# Make the wire stand out more
def morphological_wire(img):
    kernel = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    return closed

# Remove the wire
def morphological_no_wire(img):
    # kernel = np.ones((2,2), np.uint8)
    # openned = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    return img

# This is an operation used on 
def segmented_and_operation(image, kernel):
    """
    Perform an AND operation between a kernel and an image.
    If all the overlapping pixels are 255, set the center pixel to 0.

    :param image: Input image (binary)
    :param kernel: Input kernel (binary)
    :return: Processed image
    """

    # Perform the AND operation between the image and the kernel
    and_result = cv2.filter2D(image, -1, kernel)

    # Find locations where all overlapping pixels are 255
    kernel_sum = np.sum(kernel)
    locations = np.where(and_result == kernel_sum)

    # Set the center pixel to 0 in those locations
    result_image = np.copy(image)
    center_y, center_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    result_image[locations[0] - center_y, locations[1] - center_x] = 0

    return result_image



    

# Define the location of the image files
image_dir_path = "X:/Programs/Research_Project/TestImages"
output_dir_path = "X:/Programs/Research_Project/Output"

# Run through the images in the file and apply the contrast stretch function
for filename in os.listdir(image_dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(image_dir_path, filename))
        
        # Apply a color filter
        img = green_and_red_filter(img, 1.35)
        
        # To Grayscale
        wires = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((3,3), np.uint8)
        # wires = cv2.morphologyEx(wires, cv2.MORPH_OPEN, kernel)
        
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/ColorFilter", filename), wires)
        # Inversion
        wires = np.subtract(255, wires)
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Inversion", filename), wires)
        # Extract Edges
        wire = morphological_wire(wires)
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Morphological_Wire", filename), wire)
        
        no_wire = morphological_no_wire(wires)
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Morphological_NoWire", filename), no_wire)
        
        wires = np.subtract(wire, no_wire)
        
        # Contrast stretch the images for the sake of normalizing the pixel ranges
        cv2.normalize(wires, wires, 0, 255, cv2.NORM_MINMAX)
        
        # # # Threshold pixels by average value
        # # wires = average_threshold(wires, 2.9)
        # # wires = variance_threshold(wires, 15)
        
        # wires = ImageSegment(wires).process(Pixel_Segmentation(3, 50))
        
        # Perform Adaptive thresholding
        # thresh = adaptive_threshold(wires)
        # canny = CannyEdgeDetection(gray_image, 30, 150)
        
        # Find the difference 
        # wire_diff = sliding_window_diff(wires, 5)
        
        # # if the difference isn't great enough, set it to zero.
        # wires[wire_diff < 50] = 0
        
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Morphological_diff", filename), wires)
        
        wires = Sauvola_Threshold(wires, 5, 3, 0.24)
        
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Threshold", filename), wires)
        
        # cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # avg_img = cv2.filter2D(wires, -1, cross_kernel)
        # wires[avg_img > 200] = 0
        
        # cv2.imwrite(os.path.join("X:/Programs/Research_Project/AND_Threshold", filename), wires)
        
        
        
        
# Create a function that runs on an image called pixel color filter. Compare the ratio of green vs blue in each pixel. If that pixel value exceeds a threshold parameter, set the pixel value to 0.