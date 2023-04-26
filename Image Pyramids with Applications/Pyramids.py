# 1. Create image pyramids: Gaussian Laplacian (35 points)
# 2. Implement histogram and entropy of the original image using Gaussian and Laplacian decompositions (35 
# points)
# 3. Implement an image blending procedure using your 2 images (30 points)

import cv2
import numpy as np
import os
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.special import entr
import pandas as pd


def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(levels-1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1])
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def save_gaussian_pyramids(input_folder, output_folder, levels):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        file_extension = os.path.splitext(img_name)[1].lower()

        # Only accept .jpg and .png files
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            continue

        input_path = os.path.join(input_folder, img_name)
        image = cv2.imread(input_path)
        if image is None:
            continue

        # Create the Gaussian pyramid for the current image
        gaussian_pyramid = build_gaussian_pyramid(image, levels)

        # Create a subfolder for the current image's pyramid
        image_pyramid_folder = os.path.join(output_folder, os.path.splitext(img_name)[0])
        os.makedirs(image_pyramid_folder)

        # Save each level of the Gaussian pyramid as a .png file
        for i, pyramid_level in enumerate(gaussian_pyramid):
            output_path = os.path.join(image_pyramid_folder, f"level_{i}.png")
            cv2.imwrite(output_path, pyramid_level)
            
def save_laplacian_pyramids(input_folder, output_folder, levels):
    laplacian_pyramids = []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        file_extension = os.path.splitext(img_name)[1].lower()

        # Only accept .jpg and .png files
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            continue

        input_path = os.path.join(input_folder, img_name)
        image = cv2.imread(input_path)
        if image is None:
            continue

        # Create the Gaussian pyramid for the current image
        gaussian_pyramid = build_gaussian_pyramid(image, levels)
        # Create the Laplacian pyramid for the current image
        laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
        laplacian_pyramids.append(laplacian_pyramid)

        # Create subfolders for the current image's Gaussian and Laplacian pyramids
        img_base_name = os.path.splitext(img_name)[0]
        gaussian_pyramid_folder = os.path.join(output_folder, img_base_name, "gaussian")
        laplacian_pyramid_folder = os.path.join(output_folder, img_base_name, "laplacian")
        os.makedirs(gaussian_pyramid_folder)
        os.makedirs(laplacian_pyramid_folder)

        # Save each level of the Gaussian and Laplacian pyramids as .png files
        for i, (gaussian_level, laplacian_level) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
            gaussian_output_path = os.path.join(gaussian_pyramid_folder, f"level_{i}.png")
            laplacian_output_path = os.path.join(laplacian_pyramid_folder, f"level_{i}.png")
            cv2.imwrite(gaussian_output_path, gaussian_level)
            cv2.imwrite(laplacian_output_path, laplacian_level)
            
    return laplacian_pyramids

def calculate_histograms(laplacian_pyramids):
    all_histograms = []

    for laplacian_pyramid in laplacian_pyramids:
        pyramid_histograms = []
        
        for laplacian_level in laplacian_pyramid:
            # Calculate the histogram for the current Laplacian level
            hist = cv2.calcHist([laplacian_level], [0], None, [256], [0, 256])
            pyramid_histograms.append(hist)
        
        all_histograms.append(pyramid_histograms)
    
    return all_histograms

def calculate_histogram_features(histograms):
    feature_array = []

    for pyramid_histograms in histograms:
        for hist in pyramid_histograms:
            # Normalize the histogram
            hist_normalized = hist / hist.sum()

            # Calculate the entropy
            entropy = entr(hist_normalized).sum()

            # Calculate the mean
            mean = np.mean(hist_normalized)

            # Calculate the variance
            variance = np.var(hist_normalized)

            # Calculate the skew
            skewness = skew(hist_normalized)

            # Calculate the kurtosis
            kurt = kurtosis(hist_normalized)

            # Append the features as a row to the feature_array
            feature_array.append([entropy, mean, variance, skewness, kurt])

    return np.array(feature_array)

def save_histogram_features_to_csv(feature_array, output_file):
    # Convert the feature array to a pandas DataFrame
    df = pd.DataFrame(feature_array, columns=['entropy', 'mean', 'variance', 'skew', 'kurtosis'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

def blend_images(image1, image2, mask, levels):
    mask_gaussian_pyramid = build_gaussian_pyramid(mask, levels)
    
    laplacian_pyramid1 = build_laplacian_pyramid(build_gaussian_pyramid(image1, levels))
    laplacian_pyramid2 = build_laplacian_pyramid(build_gaussian_pyramid(image2, levels))
    
    blended_laplacian_pyramid = []
    for l1, l2, gm in zip(laplacian_pyramid1, laplacian_pyramid2, mask_gaussian_pyramid):
        blended_level = l1 * gm + l2 * (1 - gm)
        blended_laplacian_pyramid.append(blended_level)
        
    blended_image = blended_laplacian_pyramid[-1]
    for level in reversed(blended_laplacian_pyramid[:-1]):
        blended_image = cv2.pyrUp(blended_image) + level
        
    return blended_image


