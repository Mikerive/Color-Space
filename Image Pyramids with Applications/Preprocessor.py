import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

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


# Load an image
image = cv2.imread("input_image.jpg")

# Compute edge directions
edge_directions = structure_tensor(image)

# Visualize edge directions
cv2.imshow("Edge Directions", edge_directions)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

def preprocessor(input_folder, output_folder):
    sobel_edge_folder = os.path.join(output_folder, "sobel_edge")
    clahe_folder = os.path.join(output_folder, "clahe")
    
    if not os.path.exists(output_folder):
        os.makedirs(sobel_edge_folder)

    if not os.path.exists(sobel_edge_folder):
        os.makedirs(sobel_edge_folder)
        
    if not os.path.exists(clahe_folder):
        os.makedirs(clahe_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_image_path)
            
            clahe_image = apply_clahe(image)
            output_image_path = os.path.join(clahe_folder, filename)
            cv2.imwrite(output_image_path, clahe_image)
            
            sobel_edge_image = sobel_edge(image)
            output_image_path = os.path.join(sobel_edge_folder, filename)
            cv2.imwrite(output_image_path, sobel_edge_image)
            
            # show_lab_channels(image)

# Example usage:
input_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/test_images"
output_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/preprocessor"
preprocessor(input_folder, output_folder)
