import numpy as np
import cv2
import csv
from scipy.stats import skew, kurtosis, entropy
import os
import mahotas as mh
from skimage.util import view_as_windows
from skimage.feature import local_binary_pattern


def remove_lone_pixels(image):
    # Define kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Expand image with zero padding
    expanded = np.pad(image, 1, mode="constant")
    
    # Create sliding window view of image
    windows = view_as_windows(expanded, (3, 3))
    
    # Convolve windows with kernel
    neighbors_array = np.sum(windows * kernel, axis=(2, 3))
    
    # Identify lone pixels
    lone_pixels = (neighbors_array == 0) & image
    
    # Remove lone pixels
    output = np.multiply(image, np.logical_not(lone_pixels))
    
    return output

def Sauvola_Threshold(img, kernel_size, k=0.34, R=128, stride=1):
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

def sliding_window_diff(img, kernel_size, threshold, stride = 1):
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

    return img[diffs > threshold].astype(np.uint8)

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
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
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

# Box Counting

def box_counting(image, scale_step=0.5, threshold=None):
    """
    Perform box counting on an image using Mahotas.

    Parameters
    ----------
    image : numpy array
        The image to be analyzed.
    scale_step : float, optional
        The step size to use when scaling the image. Default is 0.5.
    threshold : float, optional
        The threshold value to use when binarizing the image. If None, the image will not be binarized.
    
    Returns
    -------
    counts : numpy array
        The box counts for each scale factor.
    scales : numpy array
        The scale factors used.
    """
    
    # Ensure input parameters are valid
    assert isinstance(image,np.ndarray), 'Input Image is not a numpy array'
    assert isinstance(scale_step,float), 'Scale_step is expected to be float'
    assert threshold is None or isinstance(threshold, float), 'Threshold is expected to be a float or None'
    
    # Helper function for scaling and counting
    def scale_and_count(image: np.ndarray, scale: float=0.1, threshold = None):
        """
        Returns the count of non-zero pixel values in the scaled image.
        
        Parameters:
        -----------
        image: numpy array
            The input image
        scale: numpy array
            The scaling factor
        threshold : float, optional
            The threshold value to use when binarizing the image.
            
        Returns:
        --------
        count: int
            The count of non-zero pixel values in the scaled image.
        """
       # Get the new dimensions of the image
        height = int(image.shape[0] * scale)
        width = int(image.shape[1] * scale)
        dim = (width, height)

        # Resize the image
        scaled_image = cv2.resize(image, dim)
            
        # Sum up the non-zero pixels and return the result
        count = cv2.countNonZero(scaled_image)
        return count
    
    
    # Initialize variables
    count_list = []
    scale_list = []
    
    # Loop over the range of scale factors
    for i in np.arange(scale_step, 1+scale_step, scale_step):
        # Call the helper function to scale and count the image
        count = scale_and_count(image=image, scale=i, threshold=threshold)
        # Append the count and scale factor to their respective lists
        count_list.append(count)
        scale_list.append(i)
    
    # Convert lists to numpy arrays
    counts = np.array(count_list)
    scales = np.array(scale_list)
    
    # Fit a line to the log-log plot of counts vs. scales
    p = np.polyfit(np.log(scales), np.log(counts), 1)
    line = np.exp(p[1]) * scales**p[0]
    
    # # Plot the log-log plot of counts vs. scales and the fitted line
    # plt.loglog(scales, counts, 'bo')
    # plt.loglog(scales, line, 'r-')
    # plt.xlabel('Scale factor (log)')
    # plt.ylabel('Box count (log)')
    # plt.show()
    
    return counts, scales

def calculate_histogram_metrics(image):
    # Calculate the histogram of the image
    hist = cv2.calcHist([image], [0], None, [255], [0, 255])
    hist_norm = np.zeros_like(hist)
    # Normalize the histogram using L1 normalization
    cv2.normalize(hist,hist_norm,1.0,0.0,cv2.NORM_L1)

    # Calculate the skewness, kurtosis, entropy, and smoothness (R) of the histogram
    skewness = skew(hist_norm)
    kurtosis_val = kurtosis(hist_norm)
    entropy_val = (-(np.multiply(hist_norm, np.log(hist_norm)))).sum()
    R = (np.sum(np.square(hist_norm)) / np.sum(hist_norm)) 
    
    return skewness, kurtosis_val, entropy_val, R


# Local Binary Pattern 

def sobel_lbp(img, radius = 1, points = 8):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    n_points = points * radius
    METHOD = 'uniform'
    lbp = local_binary_pattern(sobel, n_points, radius, METHOD)
    cv2.normalize(lbp, lbp, 0, 255, cv2.NORM_MINMAX)
    return lbp

def lbp_transform(img, radius=1, points=8):
    METHOD = 'uniform'
    lbp = local_binary_pattern(img, points, radius, METHOD)
    cv2.normalize(lbp, lbp, 0, 255, cv2.NORM_MINMAX)
    if lbp.ndim == 2:
        lbp = np.expand_dims(lbp, axis=2)
        
    return lbp.astype(np.uint8)


# Freeman Chain Code Calculations

def calculate_direction(x_diff, y_diff):
    if x_diff == 0 and y_diff == -1:
        return 0
    elif x_diff == 1 and y_diff == -1:
        return 1
    elif x_diff == 1 and y_diff == 0:
        return 2
    elif x_diff == 1 and y_diff == 1:
        return 3
    elif x_diff == 0 and y_diff == 1:
        return 4
    elif x_diff == -1 and y_diff == 1:
        return 5
    elif x_diff == -1 and y_diff == 0:
        return 6
    elif x_diff == -1 and y_diff == -1:
        return 7

def freeman_chain_code(contour):
    chain_code = []
    for i in range(len(contour) - 1):
        x_diff = contour[i][0][0] - contour[i + 1][0][0]
        y_diff = contour[i][0][1] - contour[i + 1][0][1]
        direction = calculate_direction(x_diff, y_diff)
        chain_code.append(direction)
    return chain_code

def find_max_contour(contours):
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def FCC_MaxContour(img):
    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the maximum area
    max_contour = find_max_contour(contours)

    # Draw the contour on the image with a green color and thickness of 3
    img_with_contour = cv2.drawContours(img.copy(), [max_contour], 0, (0, 255, 0), 3)

    # Get the Freeman chain code approximation of the contour
    fcc = freeman_chain_code(max_contour)

    # Return the Freeman chain code approximation
    return fcc, img_with_contour

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

# Define the location of the image files
image_dir_path = 'C:/Programs/Image Processing/Color Space/FeatureExtraction/test_images'

        
for filename in os.listdir(image_dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(image_dir_path, filename))
        
        # Apply a color filter
        img = green_and_red_filter(img, 1.5)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/ColorFilter", filename), img)
        
        # To Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Inversion
        inv = np.subtract(255, gray)
        
        # Gamma Correction
        inv = gamma_correction(inv, 0.5)
        
        # Contrast stretch the images for the sake of normalizing the pixel ranges
        cv2.normalize(inv, inv, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Inverse", filename), inv)
        # Extract Edges
        wire = morphological_wire(inv)
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Morphological_Wire", filename), wire)
        
        no_wire = morphological_no_wire(inv)
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Morphological_NoWire", filename), no_wire)
        
        diff = np.subtract(wire, no_wire)
        
        # Find the difference 
        wire_diff = sliding_window_diff(diff, 3, 50)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Morphological_diff", filename), wire_diff)
        # Sauvola is designed to threshold text in images
        inv = np.subtract(255, inv)
        
        wires = Sauvola_Threshold(inv, 5, 0.2, 128)
        # re-invert the output of sauvola thresholding
        wires = np.subtract(255, wires)
        
        wires = remove_lone_pixels(wires)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Threshold", filename), wires)
        
        lbp_img = lbp_transform(wires, 3)
        
        lbp_img = np.subtract(255, lbp_img)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/LBP", filename), lbp_img)
        
        lbp_sobel_img = sobel_lbp(wires, 3)
        
        lbp_sobel_img = np.subtract(255, lbp_sobel_img)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/Sobel_LBP", filename), lbp_sobel_img)
        
        fcc, contour_img = FCC_MaxContour(wires)
        
        cv2.imwrite(os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/FCC_MaxContour", filename), contour_img)
        
        count, scale = box_counting(wires, 0.2)
        
        skew_val, kurtosis_val, entropy_val, R_val = calculate_histogram_metrics(wires)
        
        s_count, s_scale = box_counting(lbp_img, 0.2)
        
        s_skew_val, s_kurtosis_val, s_entropy_val, s_R_val = calculate_histogram_metrics(lbp_img)
        
        
        # Pass quantitative values to a csv file.
        
        # create a list to hold the variable names
        headers = ['filename', 'kurtosis', 'entropy', 'R-value', 'skewness', 'count', 'scale']
        # create a list to hold the variable values
        values = [filename, kurtosis_val, entropy_val, R_val, skew_val, count, scale]
        lbp_values = [filename, s_kurtosis_val, s_entropy_val, s_R_val, s_skew_val, s_count, s_scale]
        
        name, ext = os.path.splitext(filename)

        file_path = os.path.join("C:/Programs/Image Processing/Color Space/FeatureExtraction/FeatureExtraction", f'{name}.csv')
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow(values)
            writer.writerow(lbp_values)
            writer.writerow(fcc)
                