import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import cv2
import numpy as np
from skimage import feature
from skimage.util import view_as_windows
import random
from scipy.stats import kurtosis, skew
import pandas as pd
from skimage.filters import threshold_sauvola
from scipy import ndimage

""" Used to produce images for presentations and test image transformations before being passed to the SVM."""

def Sauvola_Threshold(img, kernel_size = 35, k=0.28, R=50, stride=1):
    img = np.subtract(255, img)
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
    
    output_image = np.subtract(255, output_image)

    return output_image.astype(np.uint8)

def lbp_directional_segmentation(image, cell_size=(16, 16)):
    def lbp(image, radius=1, points=8):
        return feature.local_binary_pattern(image, points, radius, method='uniform')

    def lbp_histogram(image, cell_size=(16, 16), points=8):
        height, width = image.shape
        y_cells, x_cells = height // cell_size[0], width // cell_size[1]
        histogram = np.zeros((y_cells, x_cells, points+1), dtype=np.float32)
        
        for i in range(y_cells):
            for j in range(x_cells):
                cell = image[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
                hist, _ = np.histogram(cell, bins=np.arange(0, points + 2), density=True)
                histogram[i, j, :] = hist
        
        return histogram.reshape(-1, points+1)

    def max_histogram_direction(hist):
        global_hist = np.sum(hist, axis=0)
        max_direction = np.argmax(global_hist)
        return max_direction

    def segment_image(lbp_img, direction, segmented_value=255):
        segmented = np.where(lbp_img == direction, segmented_value, 0).astype(np.uint8)
        return segmented
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply LBP transformation
    lbp_image = lbp(image)

    # Compute LBP histogram
    hist = lbp_histogram(lbp_image, cell_size=cell_size)

    # Find the most common direction
    direction = max_histogram_direction(hist)

    # Segment the image
    segmented_image = segment_image(lbp_image, direction)

    return segmented_image

def structure_tensor(image, window_size=5, sigma=1):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients using Sobel operator
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    M = np.sqrt(Gx**2 + Gy**2)
    Theta = np.arctan2(Gy, Gx)
    
    # Calculate structure tensor components
    J11 = M**2 * np.cos(Theta)**2
    J12 = M**2 * np.cos(Theta) * np.sin(Theta)
    J22 = M**2 * np.sin(Theta)**2
    
    # Smooth structure tensor components using Gaussian filter
    J11_smooth = cv2.GaussianBlur(J11, (window_size, window_size), sigma)
    J12_smooth = cv2.GaussianBlur(J12, (window_size, window_size), sigma)
    J22_smooth = cv2.GaussianBlur(J22, (window_size, window_size), sigma)
    
    # Calculate edge directions using smoothed structure tensor components (vectorized operations)
    J = np.stack((J11_smooth, J12_smooth, J12_smooth, J22_smooth), axis=-1).reshape(-1, 2, 2)
    eigenvalues, eigenvectors = np.linalg.eig(J)
    max_eigenvalue_indices = np.argmax(eigenvalues, axis=-1)
    dominant_directions = eigenvectors[np.arange(J.shape[0]), max_eigenvalue_indices]
    edge_directions = np.arctan2(dominant_directions[:, 1], dominant_directions[:, 0]) * 180 / np.pi
    edge_directions = edge_directions.reshape(gray.shape)

    return edge_directions

def show_lab_channels(rgb_image, window_size=(300, 300)):
    """
    Convert an RGB image to the LAB color space and display the L, A, and B channels.

    Args:
    rgb_image (numpy.ndarray): Input image in RGB format
    window_size (tuple): Tuple of (width, height) to resize the displayed windows
    """
    
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Convert the input image from RGB to BGR, as OpenCV uses BGR by default
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Convert the BGR image to LAB color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its individual channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Normalize the L, A, and B channels to a range of [0, 255] for display purposes
    l_normalized = cv2.normalize(l_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    a_normalized = cv2.normalize(a_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    b_normalized = cv2.normalize(b_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Resize the L, A, and B channels to the specified window size
    l_resized = cv2.resize(l_normalized, window_size, interpolation=cv2.INTER_AREA)
    a_resized = cv2.resize(a_normalized, window_size, interpolation=cv2.INTER_AREA)
    b_resized = cv2.resize(b_normalized, window_size, interpolation=cv2.INTER_AREA)

    # Display the L, A, and B channels
    cv2.namedWindow('L Channel', cv2.WINDOW_NORMAL)
    cv2.namedWindow('A Channel', cv2.WINDOW_NORMAL)
    cv2.namedWindow('B Channel', cv2.WINDOW_NORMAL)

    # Set window size
    cv2.resizeWindow('L Channel', *window_size)
    cv2.resizeWindow('A Channel', *window_size)
    cv2.resizeWindow('B Channel', *window_size)

    # Calculate the window positions
    x_offset = (screen_width - window_size[0] * 3) // 2
    y_offset = (screen_height - window_size[1]) // 2

    # Move the windows to adjacent and centered positions
    cv2.moveWindow('L Channel', x_offset, y_offset)
    cv2.moveWindow('A Channel', x_offset + window_size[0], y_offset)
    cv2.moveWindow('B Channel', x_offset + window_size[0] * 2, y_offset)

    # Show the images
    cv2.imshow('L Channel', l_resized)
    cv2.imshow('A Channel', a_resized)
    cv2.imshow('B Channel', b_resized)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('0'):
            break

    cv2.destroyAllWindows()

def sobel_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to an input image using OpenCV.
    
    Args:
    image (numpy.ndarray): Input image (grayscale or color)
    clip_limit (float): Threshold for contrast limiting (default: 2.0)
    tile_grid_size (tuple of int): Size of the grid for histogram equalization (default: (8, 8))

    Returns:
    numpy.ndarray: Image with improved contrast using CLAHE
    """

    # If the input image is in color (3 channels), convert it to the Lab color space
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
    else:
        l_channel = image

    # Create a CLAHE object with the specified parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the L channel (or grayscale image)
    l_channel_clahe = clahe.apply(l_channel)

    if len(image.shape) == 3 and image.shape[2] == 3:
        # Merge the enhanced L channel back with the other channels
        lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert the Lab image back to BGR
        result_image = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
    else:
        result_image = l_channel_clahe

    return result_image

def downsample_image(image, factor):
    """
    Downsample an image by a specific factor.

    Parameters:
    image (numpy.ndarray): The image to downsample.
    factor (float): The factor by which to downsample the image. 
                    This should be a value greater than 0 and less than or equal to 1. 
                    A factor of 1 will keep the image at its original size, 
                    while a factor of 0.5 will reduce the image size by half.

    Returns:
    numpy.ndarray: The downsampled image.
    """
    if not 0 < factor <= 1:
        raise ValueError("Factor should be a value greater than 0 and less than or equal to 1")

    # Calculate the new size of the image
    new_size = (int(image.shape[1] * factor), int(image.shape[0] * factor))

    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

    return downsampled_image

def remove_small_clusters(image, min_size):
    # label image
    labeled_array, num_features = ndimage.label(image)
    # get counts of each unique label (excluding 0)
    unique, counts = np.unique(labeled_array[labeled_array>0], return_counts=True)
    # create a boolean mask where True indicates a large cluster
    large_clusters = np.isin(labeled_array, unique[counts>=min_size])
    # create a new image where only the large clusters are visible
    large_cluster_image = np.zeros_like(image)
    large_cluster_image[large_clusters] = image[large_clusters]
    return large_cluster_image

def line_contour_features(image, ratio_threshold = 1, min_size_ratio = 0.01):
    """Find line contours in an image. Filter by size, then filter by the ratio
    between the major and minor axis. Finally, utilize angle information to calculate
    the variance in the angles of powerlines in the image.
    """

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the output image and counter for qualified contours
    image_with_contours = np.stack((image,) * 3, axis=-1)
    counter = 0

    # Calculate the minimum size of the contours
    min_size = min_size_ratio * image.shape[0] * image.shape[1]
    
    # List to store the angles of the ellipses
    angles = []
    
    # List to store the areas of the contours
    areas = []

    for cnt in contours:
        # We need at least 5 points to fit the ellipse
        if len(cnt) >= 5:
            # Check if the contour is large enough
            area = cv2.contourArea(cnt)
            
            if area >= min_size:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse

                # Calculate the ratio of the major and minor axes
                ratio = max(MA, ma) / min(MA, ma)

                # Draw the contour if the ratio is above the threshold
                if ratio >= ratio_threshold:
                    # Generate a random color for the contour
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.drawContours(image_with_contours, [cnt], 0, color, 3)
                    counter += 1
                    
                    # Add the angle of the ellipse to the list
                    angles.append(angle)
                    
                    # Add the area of the contour to the list
                    areas.append(area)

    
    return image_with_contours, angles, areas

def calculate_circular_features(data):
    """
    This function takes a list of angles in degrees as input and returns various circular statistical features. 

    Parameters:
    data (list of float): The input angles in degrees.

    Returns:
    tuple: The circular mean (in degrees), circular variance, circular standard deviation, circular skewness, circular kurtosis, and roundness.
    """
    # Convert the data to radians for calculation
    data_rad = np.deg2rad(data)

    # Calculate the circular mean of the angles. This is the average direction of the data.
    mean_angle = np.arctan2(np.mean(np.sin(data_rad)), np.mean(np.cos(data_rad)))

    # Calculate the circular variance. This is a measure of the dispersion of the angles around the circular mean.
    # It takes values between 0 and 1, with 0 indicating that all angles are the same and 1 indicating maximum dispersion.
    cvar = 1 - np.sqrt(np.mean(np.sin(data_rad - mean_angle))**2 + np.mean(np.cos(data_rad - mean_angle))**2)

    # Calculate the circular standard deviation. This is another measure of dispersion that is more comparable to the linear standard deviation.
    cstd = np.sqrt(-2 * np.log(1 - cvar))

    # Calculate differences between data and mean angle for skewness and kurtosis calculations
    diff = np.arctan2(np.sin(data_rad - mean_angle), np.cos(data_rad - mean_angle))

    # Calculate the circular skewness. This is a measure of the asymmetry of the angles around the circular mean.
    # The skewness is typically between -1 and 1, with 0 indicating a symmetric distribution of angles.
    cskewness = np.mean(diff**3) / (np.mean(diff**2))**(3/2)

    # Calculate the circular kurtosis. This is a measure of the "tailedness" of the angles.
    # A kurtosis of 0 indicates a distribution with similar kurtosis to the circular uniform distribution,
    # positive values indicate a leptokurtic distribution (more peaked than the circular uniform distribution),
    # and negative values indicate a platykurtic distribution (less peaked than the circular uniform distribution).
    ckurtosis = np.mean(diff**4) / (np.mean(diff**2))**2 - 3

    # Convert the circular mean back to degrees for the output
    mean_angle = np.rad2deg(mean_angle)

    return mean_angle, cvar, cstd, cskewness, ckurtosis

def calculate_linear_features(data):
    # Calculate the mean
    mean = np.mean(data)

    # Calculate the variance
    variance = np.var(data)

    # Calculate the standard deviation
    std_dev = np.std(data)

    # Calculate the skewness
    skewness = skew(data)

    # Calculate the kurtosis
    kurt = kurtosis(data)

    return mean, variance, std_dev, skewness, kurt

def detect_lines(image, rho, theta, threshold, minLineLength, maxLineGap):
    
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize = 3)
    
    # Use HoughLinesP to detect lines in the image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    # Create a copy of the original image to draw lines on
    line_image = np.stack((image,) * 3, axis=-1)
    
    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(line_image, (x1, y1), (x2, y2), color, 2)
    
    return line_image

def sobel_grad_mag(image, kernel_size = 5):
    # Calculate Sobel gradients in the x and y directions
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    # Calculate the gradient magnitude and direction
    sobel_mag = np.hypot(sobelx, sobely)
    sobel_dir = np.arctan2(sobely, sobelx) * (180 / np.pi)
    
    # Normalize to range 0 - 255 and convert to uint8
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    return sobel_mag, sobel_dir
        
def apply_mask(original, mask):
    # Takes a segmented image as a mask
    
    original = original.astype(np.float32)
    
    if original.ndim == 3:
        if original.shape[2] == 3:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
    # Get the dimensions of both images
    original_height, original_width = original.shape[:2]
    segmented_height, segmented_width = mask.shape[:2]
    
    # Calculate the minimum width and height
    min_width = min(original_width, segmented_width)
    min_height = min(original_height, segmented_height)
    
    # Resize both images to the size of the smaller one
    original = cv2.resize(original, (min_width, min_height))
    mask = cv2.resize(mask, (min_width, min_height))
    
    # Use the mask to mask the Sobel gradient and direction
    mask = mask.astype(bool)
    masked_img = np.where(mask, original, 0)

    return masked_img

def lbp(image, points = 8, radius = 1):
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Number of points in the circularly symmetric neighbor set
    points = points
    # Radius of circle
    radius = radius

    # Calculate the LBP
    lbp = feature.local_binary_pattern(image, points, radius, method="uniform")

    return lbp

def contrast_stretch(image):
    # Perform contrast stretching
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
    stretched_image = stretched_image.astype(np.uint8)
    return stretched_image

def binary_threshold(image):
    # Convert the image to grayscale if it has multiple channels
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
                    

def preprocessor(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    sauvola_thresh_folder = os.path.join(output_folder, "sauvola_thresh")
    os.makedirs(sauvola_thresh_folder, exist_ok=True)
    clahe_folder = os.path.join(output_folder, "clahe")
    os.makedirs(clahe_folder, exist_ok=True)
    tensor_folder = os.path.join(output_folder, "tensor")
    os.makedirs(tensor_folder, exist_ok=True)
    lbp_directions_seg_folder = os.path.join(output_folder, "lbpdirseg")
    os.makedirs(lbp_directions_seg_folder, exist_ok=True)
    line_contour_folder = os.path.join(output_folder, "contours")
    os.makedirs(line_contour_folder, exist_ok=True)
    line_folder = os.path.join(output_folder, "lines")
    os.makedirs(line_folder, exist_ok=True)
    sobel_mag_folder = os.path.join(output_folder, "sobel_magnitude")
    os.makedirs(sobel_mag_folder, exist_ok=True)
    sobel_dir_folder = os.path.join(output_folder, "sobel_direction")
    os.makedirs(sobel_dir_folder, exist_ok=True)
    gradient_folder = os.path.join(output_folder, "edge_magnitude")
    os.makedirs(gradient_folder, exist_ok=True)
    directional_folder = os.path.join(output_folder, "edge_direction")
    os.makedirs(directional_folder, exist_ok=True)    
    lbp_folder = os.path.join(output_folder, "lbp")
    os.makedirs(lbp_folder, exist_ok=True)
    lbp_edge_folder = os.path.join(output_folder, "lbp_edge")
    os.makedirs(lbp_edge_folder, exist_ok=True)
    binary_thresh_folder = os.path.join(output_folder, "binary_thresh")
    os.makedirs(binary_thresh_folder, exist_ok=True)
    gamma_enhanced_folder = os.path.join(output_folder, "gamma_enhanced")
    os.makedirs(gamma_enhanced_folder, exist_ok=True)

    # Initialize an empty DataFrame to store the features for all images
    all_features = pd.DataFrame()

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_image_path)
            
            # Downsample the image by a factor
            image = downsample_image(image, 0.2)
            
            # Apply Clahe Image enhancement
            clahe_image = apply_clahe(image)
            output_image_path = os.path.join(clahe_folder, filename)
            cv2.imwrite(output_image_path, clahe_image)
            
            gamma_image = adjust_gamma(clahe_image, 0.2)
            gamma_image = gamma_image.astype(np.uint8)
            output_image_path = os.path.join(gamma_enhanced_folder, filename)
            cv2.imwrite(output_image_path, gamma_image)
            
            # Convert to gray, apply Sauvola_Threshold, then remove small clusters.
            segmented_image = remove_small_clusters(Sauvola_Threshold(adjust_gamma(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY), 0.2)), 50)
            segmented_image = segmented_image.astype(np.uint8)
            output_image_path = os.path.join(sauvola_thresh_folder, filename)
            cv2.imwrite(output_image_path, segmented_image)
            
            # Attempt binary threshold
            binary_thresh = remove_small_clusters(adjust_gamma(binary_threshold(clahe_image), 2), 50)
            output_image_path = os.path.join(binary_thresh_folder, filename)
            cv2.imwrite(output_image_path, binary_thresh)
            
            # Probabilistic Hough Line detection
            line_image = detect_lines(segmented_image, 1, np.pi/180, 100, 100, 10)
            output_image_path = os.path.join(line_folder, filename)
            cv2.imwrite(output_image_path, line_image)
            
            # Produce sobel magnitude and direction image representations
            mag_img, dir_img = sobel_grad_mag(clahe_image, 7)
            
            output_image_path = os.path.join(sobel_mag_folder, filename)
            cv2.imwrite(output_image_path, mag_img)
            output_image_path = os.path.join(sobel_dir_folder, filename)
            cv2.imwrite(output_image_path, dir_img)
            
            # Produce lbp image representation
            lbp_img = np.subtract(255, contrast_stretch(lbp(clahe_image, 24, 8)))
            output_image_path = os.path.join(lbp_folder, filename)
            cv2.imwrite(output_image_path, lbp_img)
            
            # Segmented sobel magnitude and direction
            mag_thresh = apply_mask(mag_img, segmented_image)
            dir_thresh = apply_mask(dir_img, segmented_image)
            
            output_image_path = os.path.join(gradient_folder, filename)
            cv2.imwrite(output_image_path, mag_thresh)
            output_image_path = os.path.join(directional_folder, filename)
            cv2.imwrite(output_image_path, dir_thresh)
            
            # lbp edge
            lbp_thresh = apply_mask(lbp_img, segmented_image)
            output_image_path = os.path.join(lbp_edge_folder, filename)
            cv2.imwrite(output_image_path, lbp_thresh)
            
            # lbp_directions_seg = lbp_directional_segmentation(image)
            # output_image_path = os.path.join(lbp_directions_seg_folder, filename)
            # cv2.imwrite(output_image_path, lbp_directions_seg)
            
            # Apply contour analysis
            contours, angles, areas = line_contour_features(segmented_image)
            
            output_image_path = os.path.join(line_contour_folder, filename)
            cv2.imwrite(output_image_path, contours)
            # Calculate circular features for angles
            circular_features = calculate_circular_features(angles)
            # Calculate linear features for areas
            linear_features = calculate_linear_features(areas)

            # Combine features into a single list
            combined_features = list(circular_features) + list(linear_features)

            # Create a DataFrame for the features
            df = pd.DataFrame([combined_features], columns=['mean_angle', 'circular_variance', 'circular_std_dev', 
                                                            'circular_skewness', 'circular_kurtosis', 'mean_area', 
                                                            'variance_area', 'std_dev_area', 'skewness_area', 
                                                            'kurtosis_area'])

            # Add the image name to the DataFrame
            df['image_name'] = filename

            # Append the features of the current image to the DataFrame of all features
            all_features = pd.concat([all_features, df], ignore_index=True)

    # Combine the directory path and the filename
    csv_path = os.path.join(output_folder, 'features.csv')
    # Write the DataFrame to a CSV file
    all_features.to_csv(csv_path, index=False)

# Example usage:
input_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/preprocessor_images"
output_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/preprocessor"
preprocessor(input_folder, output_folder)
