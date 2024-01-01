import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.special import entr
from sklearn.impute import SimpleImputer
import pickle
from skimage.util import view_as_windows
from scipy import ndimage
from skimage.measure import label
from skimage import feature



# SVM Image Classifier
class ImageClassifier:
    # Requires svm folder, test, and train folders to be filled with classes 0 and 1.
    def __init__(self, present_folder, levels):
        self.levels = levels
        self.classes = ['0', '1']
        
        self.svm_folder = os.path.join(present_folder, "svm")
        os.makedirs(self.svm_folder, exist_ok=True)

        self.train_folder = os.path.join(self.svm_folder, "train")
        os.makedirs(self.train_folder, exist_ok=True)

        self.test_folder = os.path.join(self.svm_folder, "test")
        os.makedirs(self.test_folder, exist_ok=True)

        self.output_csv = os.path.join(self.svm_folder, "predictions.csv")
        self.train_features_csv = os.path.join(self.svm_folder, "train_features.csv")
        self.classifications_csv = os.path.join(self.svm_folder, "classifications.csv")

    def build_gaussian_pyramid(self, image, levels):
        gaussian_pyramid = [image]
        for _ in range(levels-1):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)
        return gaussian_pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid):
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
    
    def nonzero_pyramidal_histograms(self, laplacian_pyramids, num_bins=8):
        "This histogram calculates the nonzero value pyramidal histograms for pyramidal decompositions"
        all_histograms = []

        for laplacian_pyramid in laplacian_pyramids:
            pyramid_histograms = []
            
            for laplacian_level in laplacian_pyramid:
                # Calculate the histogram for the current Laplacian level
                hist, _ = np.histogram(laplacian_level, bins=num_bins, range=(1, num_bins + 1))
                pyramid_histograms.append(hist)
            
            all_histograms.append(pyramid_histograms)
        
        return all_histograms

    def calculate_histogram_features(self, histograms):
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
                skew_result = skew(hist_normalized)
                skewness = skew_result[0] if isinstance(skew_result, np.ndarray) else skew_result

                # Calculate the kurtosis
                kurt_result = kurtosis(hist_normalized)
                kurt = kurt_result[0] if isinstance(kurt_result, np.ndarray) else kurt_result

                # Append the features as a row to the feature_array
                feature_array.append([entropy, mean, variance, skewness, kurt])

        return np.array(feature_array)

    def create_data(self, folder, train):
        X = []
        y = []
        
        processing_folder = ''
        
        # Create the processing folder and subfolders for each function output
        if train == 1:
            processing_folder = os.path.join(self.svm_folder, "train")
            os.makedirs(processing_folder, exist_ok=True)
        if train == 0:
            processing_folder = os.path.join(self.svm_folder, "test")
            os.makedirs(processing_folder, exist_ok=True)

        segmentation_folder = os.path.join(processing_folder, "segmentation")
        os.makedirs(segmentation_folder, exist_ok=True)

        lbp_folder = os.path.join(processing_folder, "lbp")
        os.makedirs(lbp_folder, exist_ok=True)

        dir_folder = os.path.join(processing_folder, "dir")
        os.makedirs(dir_folder, exist_ok=True)
        
        mag_folder = os.path.join(processing_folder, "mag")
        os.makedirs(mag_folder, exist_ok=True)
        
        img_counter = 1
        
        def downsample_image(image : np.ndarray, factor):
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
        
        def Sauvola_Threshold(img, kernel_size = 35, k=0.28, R=128, stride=1):
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

        def contrast_stretch(image):
            # Perform contrast stretching
            min_val = np.min(image)
            max_val = np.max(image)
            stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
            stretched_image = stretched_image.astype(np.uint8)
            return stretched_image

        def adjust_gamma(image, gamma=1.0):
            # build a lookup table mapping the pixel values [0, 255] to
            # their adjusted gamma values
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
            # apply gamma correction using the lookup table
            return cv2.LUT(image, table)
                   
        def morphological_closing(image, kernel_size=(25,25)):
            # Create an elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

            # Perform the morphological closing operation
            closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            return closed_image
 
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

        row_vector = np.array([])
        
        def feature_appender(masked_img):
            nonlocal row_vector
            gaussian_pyramid = self.build_gaussian_pyramid(masked_img, self.levels)
            laplacian_pyramid = self.build_laplacian_pyramid(gaussian_pyramid)
            
            euler_numbers = []
            for img in laplacian_pyramid:
                # Convert the image to binary
                binary_img = img > 0

                # Calculate the Euler number for the binary image
                labeled_img, num = label(binary_img, return_num=True)
                euler_number = num - len(np.unique(labeled_img))
                euler_numbers.append(euler_number)


            gaussian_histograms = self.nonzero_pyramidal_histograms([gaussian_pyramid])
            laplacian_histograms = self.nonzero_pyramidal_histograms([laplacian_pyramid])

            gaussian_features = self.calculate_histogram_features(gaussian_histograms)
            laplacian_features = self.calculate_histogram_features(laplacian_histograms)

            # Ravel Gaussian and Laplacian features
            gaussian_features_raveled = np.ravel(gaussian_features)
            laplacian_features_raveled = np.ravel(laplacian_features)

            # Concatenate Gaussian and Laplacian features
            features = np.concatenate((gaussian_features_raveled, laplacian_features_raveled), axis=0)
            
            # Append Euler numbers
            features = np.concatenate((features, euler_numbers), axis=0)

            # Concatenate the features to the row_vector
            row_vector = np.concatenate((row_vector, features), axis=0)
        
        for class_label in self.classes:
            class_folder = os.path.join(folder, class_label)
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_folder, img_name)
                    img = cv2.imread(img_path)
                    
                    img = downsample_image(img, 0.1)
                    
                    clahe_image = apply_clahe(img)
                    
                    # Save segmentation output
                    binary_map = remove_small_clusters(Sauvola_Threshold(adjust_gamma(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY), 0.2)), 50)
                    cv2.imwrite(os.path.join(segmentation_folder, img_name), contrast_stretch(binary_map))
                    
                    
                    # Apply morphological closing to the binary map
                    binary_map = morphological_closing(binary_map, kernel_size=(5,5))
                    
                    # Save LBP output
                    lbp_img = lbp(clahe_image, 24, 8)
                    lbp_thresh = apply_mask(clahe_image, binary_map)
                    cv2.imwrite(os.path.join(lbp_folder, f"{img_counter}.png"), lbp_img)
                    
                    mag_img, dir_img = sobel_grad_mag(clahe_image, 3)

                    # Save masked output
                    mag_thresh = apply_mask(mag_img, binary_map)
                    dir_thresh = apply_mask(dir_img, binary_map)
                    cv2.imwrite(os.path.join(mag_folder, img_name), mag_thresh)
                    cv2.imwrite(os.path.join(dir_folder, img_name), dir_thresh)
                
                    feature_appender(mag_img)
                    feature_appender(dir_img)
                    feature_appender(lbp_thresh)
                    # feature_appender(img[:,:,1])
                    # feature_appender(img[:,:,2])
                    # feature_appender(img[:,:,0])
                    
                    # Transpose summed_features to make it a row vector
                    row_features = np.reshape(row_vector, (1, -1))
                    X.extend(row_features)
                    
                    # Reset row_vector for each image
                    row_vector = np.array([])
                    
                    y.append(class_label)
                    
                    img_counter += 1
        
        return X, y

    def train(self):
        X_train, y_train = self.create_data(self.train_folder, 1)

        # Impute missing values
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer.fit(X_train)
        X_train = self.imputer.transform(X_train)
        print("...imputing")
        
        # Save training features to a CSV file
        train_features_df = pd.DataFrame(X_train)
        train_features_df["label"] = y_train
        train_features_df.to_csv(self.train_features_csv, index=False)
        print("...training")

        # Train SVM
        self.clf = SVC(kernel='linear')
        self.clf.fit(X_train, y_train)
        
        # Save the trained model to a file
        model_file = os.path.join(self.svm_folder, "svm_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(self.clf, f)

        # Save the imputer to a file
        imputer_file = os.path.join(self.svm_folder, "imputer.pkl")
        with open(imputer_file, 'wb') as f:
            pickle.dump(self.imputer, f)

        print("...model and imputer saved")
        
    def load_model(self):
        # Load the trained model from a file
        model_file = os.path.join(self.svm_folder, "svm_model.pkl")
        with open(model_file, 'rb') as f:
            self.clf = pickle.load(f)
            
        # Load the imputer from a file
        imputer_file = os.path.join(self.svm_folder, "imputer.pkl")
        with open(imputer_file, 'rb') as f:
            self.imputer = pickle.load(f)
        
        print("...model and imputer loaded")

    def test(self):
        X_test, y_test = self.create_data(self.test_folder, 0)
        
        # Impute missing values
        X_test = self.imputer.transform(X_test)
        print("...imputing")
        
        # Test SVM
        y_pred = self.clf.predict(X_test)
        print("...predictions")

        # Calculate prediction statistics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=self.classes).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'predictions': y_pred,
            'actual': y_test
        })
        df.to_csv(self.classifications_csv, index=False)

        # Save prediction statistics to a CSV file
        df = pd.DataFrame({
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1_score],
            'true_positives': [tp],
            'true_negatives': [tn],
            'false_positives': [fp],
            'false_negatives': [fn],
        })
        
        print("...output")
        df.to_csv(self.output_csv, index=False)
        
present_folder = "C:/Programs/Image Processing/Color Space/Image Pyramids with Applications/"
levels = 5

svm = ImageClassifier(present_folder, levels)
svm.train()
# svm.load_model()
svm.test()
