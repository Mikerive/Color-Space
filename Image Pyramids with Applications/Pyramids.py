# 1. Create image pyramids: Gaussian Laplacian (35 points)
# 2. Implement histogram and entropy of the original image using Gaussian and Laplacian decompositions (35 
# points)
# 3. Implement an image blending procedure using your 2 images (30 points)

import cv2
import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from skimage.feature import local_binary_pattern

# Blend Images
class Labeler:
    def __init__(self):
        self.draw_labeling = False
        self.last_x = None
        self.last_y = None
        self.img = None
        self.labels = None
        self.display_img = None
    
    def draw_line(self, x, y):
        cv2.line(self.labels, (self.last_x, self.last_y), (x, y), 1, thickness=400)
        self.last_x, self.last_y = x, y
       
    def label_image(self, event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            if self.draw_labeling:
                self.draw_line(x, y)
                
    def on_press(self, event):
        if event.xdata and event.ydata:
            self.draw_labeling = True
            self.last_x, self.last_y = int(event.xdata), int(event.ydata)

    def on_release(self, event):
        self.draw_labeling = False
        self.update_display()
        
    def update_display(self):
        self.display_img = cv2.addWeighted(self.img, 0.7, self.labels * 255, 0.3, 0)
        plt.imshow(self.display_img, cmap='gray')
        plt.draw()
        
    def get_labels(self, input_image_path):
        self.img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        self.labels = np.zeros_like(self.img, dtype=np.uint8)
        fig, ax = plt.subplots()
        plt.imshow(self.img, cmap='gray')
        cid = fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.label_image)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        return self.labels

# Pyramidization Functions
def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(levels-1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])
        
        # Adjust the dimensions of the upsampled image to match the current Gaussian pyramid level
        height, width = gaussian_pyramid[i].shape[:2]
        upsampled = cv2.resize(upsampled, (width, height))

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

def calculate_histograms(laplacian_pyramids, num_bins=8):
    all_histograms = []

    for laplacian_pyramid in laplacian_pyramids:
        pyramid_histograms = []
        
        for laplacian_level in laplacian_pyramid:
            # Calculate the histogram for the current Laplacian level
            hist, _ = np.histogram(laplacian_level[laplacian_level > 0], bins=num_bins, range=(1, num_bins + 1))
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
            skewness = skew(hist_normalized)[0]

            # Calculate the kurtosis
            kurt = kurtosis(hist_normalized)[0]

            # Append the features as a row to the feature_array
            feature_array.append([entropy, mean, variance, skewness, kurt])

    return np.array(feature_array)

def save_histogram_features_to_csv(feature_array, output_file):
    # Convert the feature array to a pandas DataFrame
    df = pd.DataFrame(feature_array, columns=['entropy', 'mean', 'variance', 'skew', 'kurtosis'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

def pyramids(input_folder, output_folder, levels):
    gaussian_output_folder = os.path.join(output_folder, "gaussian")
    laplacian_output_folder = os.path.join(output_folder, "laplacian")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(gaussian_output_folder):
        os.makedirs(gaussian_output_folder)

    if not os.path.exists(laplacian_output_folder):
        os.makedirs(laplacian_output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_image_path)
            
            # Create and save Gaussian pyramid
            gaussian_pyramid = build_gaussian_pyramid(image, levels)
            image_base_name = os.path.splitext(filename)[0]
            gaussian_pyramid_folder = os.path.join(gaussian_output_folder, image_base_name)
            if not os.path.exists(gaussian_pyramid_folder):
                os.makedirs(gaussian_pyramid_folder)
            
            for i, pyramid_level in enumerate(gaussian_pyramid):
                output_path = os.path.join(gaussian_pyramid_folder, f"level_{i}.png")
                cv2.imwrite(output_path, pyramid_level)
            
            # Create and save Laplacian pyramid
            laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
            laplacian_pyramid_folder = os.path.join(laplacian_output_folder, image_base_name)
            if not os.path.exists(laplacian_pyramid_folder):
                os.makedirs(laplacian_pyramid_folder)
            
            for i, pyramid_level in enumerate(laplacian_pyramid):
                output_path = os.path.join(laplacian_pyramid_folder, f"level_{i}.png")
                cv2.imwrite(output_path, pyramid_level)
                
            # Calculate histograms and histogram features for Gaussian pyramid
            gaussian_histograms = calculate_histograms([gaussian_pyramid])
            gaussian_features = calculate_histogram_features(gaussian_histograms)

            # Save Gaussian pyramid histogram features to a CSV file
            gaussian_csv_path = os.path.join(gaussian_pyramid_folder, "histogram_features.csv")
            save_histogram_features_to_csv(gaussian_features, gaussian_csv_path)
                
            # Calculate histograms and histogram features for Laplacian pyramid
            laplacian_histograms = calculate_histograms([laplacian_pyramid])
            laplacian_features = calculate_histogram_features(laplacian_histograms)

            # Save Laplacian pyramid histogram features to a CSV file
            laplacian_csv_path = os.path.join(laplacian_pyramid_folder, "histogram_features.csv")
            save_histogram_features_to_csv(laplacian_features, laplacian_csv_path)


# # Pyramids
# input_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/pyramid_applications/test_images"
# output_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/pyramid_applications/pyramids"
# levels = 5
# pyramids(input_folder, output_folder, levels)


# Pyramid Based - SVM
train_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/svm/train"
test_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/svm/test"
output_csv = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/svm/predictions.csv"
levels = 5

    
def blend_images(image1, image2, mask, num_levels, output_folder, blur_kernel = (135, 135)):
    
    def resize_images_to_same_dimensions(image1, image2):
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]

        target_height = min(height1, height2)
        target_width = min(width1, width2)

        # Round down to the nearest power of 2
        target_height = 2 ** int(np.log2(target_height))
        target_width = 2 ** int(np.log2(target_width))

        resized_image1 = cv2.resize(image1, (target_width, target_height))
        resized_image2 = cv2.resize(image2, (target_width, target_height))

        return resized_image1, resized_image2
    
    image1, image2 = resize_images_to_same_dimensions(image1, image2)
    
    # Resize labels to the same dimensions
    mask = cv2.resize(mask, (image1.shape[1], image1.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32)
    
    # Ensure that the input images have the same dimensions and channels
    assert image1.shape == image2.shape, "Input images must have the same dimensions and channels"
    assert image1.shape == mask.shape, "Mask must have the same dimensions and channels as the input images"

    # Step 1: Create Gaussian pyramids for both input images
    gaussian_pyramid1 = [image1.copy()]
    gaussian_pyramid2 = [image2.copy()]

    for i in range(num_levels - 1):
        gaussian_pyramid1.append(cv2.pyrDown(gaussian_pyramid1[-1]))
        gaussian_pyramid2.append(cv2.pyrDown(gaussian_pyramid2[-1]))

    # Step 2: Create Laplacian pyramids for both input images
    laplacian_pyramid1 = build_laplacian_pyramid(gaussian_pyramid1)
    laplacian_pyramid2 = build_laplacian_pyramid(gaussian_pyramid2)

    # Step 3: Create a Gaussian pyramid for the mask image
    mask_pyramid = [mask.copy()]
    for i in range(num_levels - 1):
        mask_pyramid.append(cv2.pyrDown(cv2.GaussianBlur(mask_pyramid[-1], blur_kernel, 0)))
    

    # Step 4: Blend the Laplacian pyramids using the mask pyramid
    blended_pyramid = []
    for lap1, lap2, mask_level in zip(laplacian_pyramid1, laplacian_pyramid2, mask_pyramid):
        blended_level = lap1 * mask_level + lap2 * np.subtract(1.0, mask_level)
        blended_pyramid.append(blended_level)

    # Step 5: Collapse the blended Laplacian pyramid to obtain the final blended image
    blended_image = blended_pyramid[-1]
    for i in range(num_levels - 2, -1, -1):
        upsampled = cv2.pyrUp(blended_image)
        height, width = blended_pyramid[i].shape[:2]
        upsampled = cv2.resize(upsampled, (width, height))
        blended_image = cv2.add(upsampled, blended_pyramid[i])
        
    laplacian1_output_folder = os.path.join(output_folder, "laplacian1")
    laplacian2_output_folder = os.path.join(output_folder, "laplacian2")
    mask_output_folder = os.path.join(output_folder, "mask")
    
    if not os.path.exists(laplacian1_output_folder):
        os.makedirs(laplacian1_output_folder)
    if not os.path.exists(laplacian2_output_folder):
        os.makedirs(laplacian2_output_folder)
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
        
    for i, pyramid_level in enumerate(laplacian_pyramid1):
        output_path = os.path.join(laplacian1_output_folder, f"level_{i}.png")
        cv2.imwrite(output_path, pyramid_level*255)
        
    for i, pyramid_level in enumerate(laplacian_pyramid2):
        output_path = os.path.join(laplacian2_output_folder, f"level_{i}.png")
        cv2.imwrite(output_path, pyramid_level*255)
    
    for i, pyramid_level in enumerate(mask_pyramid):
        output_path = os.path.join(mask_output_folder, f"level_{i}.png")
        cv2.imwrite(output_path, pyramid_level*255)
        
    return blended_image


# Blender
image1_path = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/blender/1.jpg"
image2_path = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/blender/2.jpg"
output_path = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/blender/blended_image.jpg"
pyramid_levels = 5

# Create a Labeler object
labeler = Labeler()

# Get the labels for the reference image
mask = labeler.get_labels(image1_path)

# Load the two images to blend
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Write the mask to a file
output_folder = f"C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/blender/"

blended_image = blend_images(image1, image2, mask, pyramid_levels, output_folder)
cv2.imwrite(output_path, blended_image)

        
