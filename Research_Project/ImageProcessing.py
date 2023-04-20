import cv2
import numpy as np
import os
from skimage.util import view_as_windows
import random
from typing import List
from sklearn.cluster import DBSCAN

# Take the grouped points, fit a curve, and plot the curves.

def plot_curves_on_image(image, points_with_labels, degree=2):
    # Create a blank image to draw the curves
    curve_image = np.zeros_like(image, dtype=np.uint8)
    
    # Extract the unique labels from the points_with_labels array
    unique_labels = np.unique(points_with_labels[:, 2])

    # Iterate through the unique labels
    for label in unique_labels:
        # Skip the noise points (label = -1)
        if label == -1:
            continue

        # Get the points corresponding to the current label
        label_points = points_with_labels[points_with_labels[:, 2] == label]

        # Extract the x and y coordinates
        x = label_points[:, 0]
        y = label_points[:, 1]

        # Fit a polynomial to the points
        coeffs = np.polyfit(x, y, degree)

        # Generate x coordinates for the curve (you can customize the step size)
        x_curve = np.arange(np.min(x), np.max(x), 1)

        # Calculate the y coordinates for the curve
        y_curve = np.polyval(coeffs, x_curve)
        
        color = tuple(np.random.randint(0, 256, 3).tolist())

        # Draw the curve on the curve_image using polylines
        curve_points = np.column_stack((x_curve, y_curve)).astype(np.int32)
        cv2.polylines(curve_image, [curve_points], False, color, 1)

    return curve_image

# Cluster Close Lines

def lines_to_points(lines, num_points = 10):
    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Linearly interpolate between the endpoints
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        
        # Combine x and y coordinates and append them to all_points list
        points = np.column_stack((xs, ys))
        all_points.append(points)

    return np.vstack(all_points)

def line_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def apply_dbscan(lines, num_points=10, eps_distance=10, min_samples=1, eps_angle=10):
    """
    Applies the DBSCAN clustering algorithm to the given set of lines, taking into account both their
    spatial proximity and their orientations.

    Parameters:
    lines (array-like): A list of lines, each represented as a nested list with coordinates (x1, y1, x2, y2).
    num_points (int, optional): The total number of points to be generated along each line, including the endpoints. Default is 10.
    eps_distance (float, optional): The maximum distance between two points for them to be considered in the same neighborhood. Default is 10.
    min_samples (int, optional): The minimum number of points required to form a dense region. Default is 1.
    eps_angle (float, optional): The maximum difference in angles between two lines for them to be considered in the same neighborhood. Default is 10.

    Returns:
    tuple: A tuple containing two elements:
        - points (numpy.ndarray): A 2D array of points generated along the lines, each row representing a point (x, y, angle).
        - labels (numpy.ndarray): A 1D array of cluster labels for each point in 'points', where -1 represents noise points.

    """
    points = lines_to_points(lines, num_points)
    angles = np.array([line_angle(line) for line in lines]).repeat(num_points).reshape(-1, 1)
    
    # Standardize angles
    angles = angles % 180
    angles[angles > 90] = angles[angles > 90] - 180

    # Create the input feature matrix for DBSCAN
    X = np.hstack((points, angles))
    
    # Define the custom distance metric
    def custom_distance_metric(a, b):
        dist_points = np.linalg.norm(a[:2] - b[:2])
        dist_angle = min(abs(a[2] - b[2]), 180 - abs(a[2] - b[2]))
        return dist_points + eps_distance * (dist_angle / eps_angle)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps_distance, min_samples=min_samples, metric=custom_distance_metric).fit(X)
    
    points = np.squeeze(points)
    
    return np.hstack((points, clustering.labels_.reshape(-1, 1)))


def cluster_lines(lines, eps_angle=10, eps_distance=10, min_samples=1):
    points = lines_to_points(lines)
    labels = apply_dbscan(points, eps_distance, min_samples, eps_angle)
    # averaged_lines = collect_endpoints(lines, labels)
    return labels

def rht_lines(image: np.ndarray, threshold: int, num_iterations: int) -> List[tuple]:
    rows, cols = image.shape
    accumulator = np.zeros((rows, cols), dtype=np.uint8)

    edges = np.argwhere(image > 0)
    for _ in range(num_iterations):
        # Randomly pick two points in the edge image
        y1, x1 = random.choice(edges)
        y2, x2 = random.choice(edges)

        # Skip iteration if points are the same
        if x1 == x2 and y1 == y2:
            continue

        # Calculate line parameters
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Increment the accumulator for the line
        ys = np.arange(rows)
        xs = np.array((ys - b) / m, dtype=np.int32)
        valid_xs = xs[(xs >= 0) & (xs < cols)]
        accumulator[ys[(xs >= 0) & (xs < cols)], valid_xs] += 1

    lines = list(zip(*np.where(accumulator > threshold)))

    return lines

def apply_random_hough_transform(image, contour):
    # Create a black image with the same dimensions as the input image
    contour_image = np.zeros_like(image)

    # Draw the contour on the black image
    cv2.drawContours(contour_image, contour, -1, 255, 1)

    # Apply the probabilistic Hough transform
    lines = cv2.HoughLinesP(contour_image, 1, np.pi / 180, threshold=8, minLineLength=8, maxLineGap=4)
    
    # Create a new color image to draw contours on
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Draw the detected lines on the contour image
    if lines is not None:
        for line in lines:
            color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color
            x1, y1, x2, y2 = line[0]
            cv2.line(contour_image, (x1, y1), (x2, y2), color, 1)

    return contour_image, lines

def find_contours(img):

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    output = np.zeros_like(img)
    # Draw contours on the original image
    return cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

def color_contours(thresholded):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new color image to draw contours on
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    

    # Draw each contour with a different color
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color
        cv2.drawContours(contour_image, [contour], -1, color, 2)
        
    return contour_image, contours
    

# Define the location of the image files
image_dir_path = "C:/Programs/Image Processing/Color Space/Research_Project/test_images"

# Run through the images in the file and apply the contrast stretch function
for filename in os.listdir(image_dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        image = cv2.imread(os.path.join(image_dir_path, filename))
        
        # To Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        contour_image, contours = color_contours(thresholded)
        
        cv2.imwrite(os.path.join('C:/Programs/Image Processing/Color Space/Research_Project/Contours', filename), contour_image)
        
        hough_image, lines= apply_random_hough_transform(image, contours)
        
        cv2.imwrite(os.path.join('C:/Programs/Image Processing/Color Space/Research_Project/RandomHough', filename), hough_image)
        
        print(lines)
        
        # Apply DBSCAN clustering to the lines
        points_with_labels = apply_dbscan(lines, eps_distance=5, min_samples=1, eps_angle=10)
        
        output = plot_curves_on_image(hough_image, points_with_labels)
        
        cv2.imwrite(os.path.join('C:/Programs/Image Processing/Color Space/Research_Project/hough', filename), output)