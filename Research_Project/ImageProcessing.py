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
import mahotas as mh


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

def green_and_red_filter(cv_image: np.ndarray, ratio: float):
    # Get the green, red, and blue values of all pixels
    bgr_values = cv_image[..., :3]

    # Compute the ratio of green vs blue for all pixels
    green_blue_ratios = bgr_values[..., 1] / bgr_values[..., 0]

    # Compute the ratio of red vs blue for all pixels
    red_blue_ratios = bgr_values[..., 2] / bgr_values[..., 0]

    # Set pixels with ratio greater than ratio to white
    white_mask = (green_blue_ratios > ratio) & (red_blue_ratios > ratio)
    cv_image[white_mask] = (255, 255, 255)

    return cv_image

def bgr_to_gray(image):
    """
    Convert cv2 image from BGR to grayscale color space.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def adaptive_threshold(image):
    """
    Apply adaptive global thresholding on a grayscale image.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return threshold_image

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# if a region of the image is too busy, remove that pixel - it isn't a wire
def average_threshold(img, threshold):
    kernel = np.ones((5,5), dtype=np.uint8)
    avg = cv2.filter2D(img, -1, kernel) // 9
    img[avg>threshold] = 0
    return img

# Morphological processing to extract edge information
def rgb_morphological_diff(img):
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    diff = np.subtract(closed, img)
    eroded_diff = cv2.erode(diff, np.ones((3,3))) 
    wires = np.subtract(diff, eroded_diff)
    
    return wires
    

# Define the location of the image files
image_dir_path = "X:/Programs/Research_Project/TestImages"
output_dir_path = "X:/Programs/Research_Project/Output"

# Run through the images in the file and apply the contrast stretch function
for filename in os.listdir(image_dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(image_dir_path, filename))
        
        wires = rgb_morphological_diff(img)
        
        wires = average_threshold(wires, 30)
        
        cv2.imwrite(os.path.join("X:/Programs/Research_Project/Inverted", filename), wires)
        
        gray_image = cv2.cvtColor(wires, cv2.COLOR_BGR2GRAY)
        # inverted_image = np.subtract(255,gray_image)
        
        
        gamma_img = gamma_correction(gray_image, 1)
        
        # cv2.imwrite(os.path.join("X:/Programs/Research_Project/Inverted", filename), median_filtered)
        
        # # Apply the Sauvola thresholding algorithm with custom parameters
        # threshold_img = cv2.adaptiveThreshold(median_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
        
        # cv2.imwrite(os.path.join(output_dir_path, filename), threshold_img)
        
# Create a function that runs on an image called pixel color filter. Compare the ratio of green vs blue in each pixel. If that pixel value exceeds a threshold parameter, set the pixel value to 0.