import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.special import entr
from sklearn.impute import SimpleImputer
from skimage import filters



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
                hist, _ = np.histogram(laplacian_level[laplacian_level > 0], bins=num_bins, range=(1, num_bins + 1))
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

    def create_data(self, folder):
        X = []
        y = []
        
        # Create the processing folder and subfolders for each function output
        processing_folder = os.path.join(self.svm_folder, "processing")
        os.makedirs(processing_folder, exist_ok=True)

        segmentation_folder = os.path.join(processing_folder, "segmentation")
        os.makedirs(segmentation_folder, exist_ok=True)

        lbp_folder = os.path.join(processing_folder, "lbp")
        os.makedirs(lbp_folder, exist_ok=True)

        masked_folder = os.path.join(processing_folder, "masked")
        os.makedirs(masked_folder, exist_ok=True)
        
        img_counter = 1
        
        def segmentation(img, window_size, k, r):
            # Apply Sauvola segmentation
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # inverted_gray_img = np.subtract(255, gray_img)
            thresh_sauvola = threshold_sauvola(gray_img, window_size=window_size, k=k, r=r)
            binary_img = gray_img > thresh_sauvola
            binary_img = np.subtract(1, binary_img)
            return binary_img.astype(np.uint8)

        def lbp(img):
            # Apply LBP
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_img = local_binary_pattern(gray_img, P=16, R=2, method='ror')
            return lbp_img
        
        def masking(img, binary_map):
            lbp_edges = cv2.bitwise_and(img, img, mask=binary_map)
            lbp_edges[binary_map == 0] = 0
            return lbp_edges
        
        sauvola_window_size=35
        sauvola_k=0.286
        sauvola_r=128
        
        def preprocessor(image):
            binary_map = segmentation(image, sauvola_window_size, sauvola_k, sauvola_r)
            lbp_img = lbp(image)
            masked = masking(lbp_img, binary_map)

            return masked
        
        row_vector = np.array([])
        
        def feature_appender(masked_img):
            nonlocal row_vector
            gaussian_pyramid = self.build_gaussian_pyramid(masked_img, self.levels)
            laplacian_pyramid = self.build_laplacian_pyramid(gaussian_pyramid)

            gaussian_histograms = self.nonzero_pyramidal_histograms([gaussian_pyramid])
            laplacian_histograms = self.nonzero_pyramidal_histograms([laplacian_pyramid])

            gaussian_features = self.calculate_histogram_features(gaussian_histograms)
            laplacian_features = self.calculate_histogram_features(laplacian_histograms)

            # Ravel Gaussian and Laplacian features
            gaussian_features_raveled = np.ravel(gaussian_features)
            laplacian_features_raveled = np.ravel(laplacian_features)

            # Concatenate Gaussian and Laplacian features
            features = np.concatenate((gaussian_features_raveled, laplacian_features_raveled), axis=0)

            # Concatenate the features to the row_vector
            row_vector = np.concatenate((row_vector, features), axis=0)
        
        for class_label in self.classes:
            class_folder = os.path.join(folder, class_label)
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_folder, img_name)
                    img = cv2.imread(img_path)
                    
                    img = cv2.resize(img, (img.shape[1], img.shape[0]))
                    
                    # Save segmentation output
                    binary_map = segmentation(img, sauvola_window_size, sauvola_k, sauvola_r)
                    cv2.imwrite(os.path.join(segmentation_folder, f"{img_counter}.png"), binary_map * 255)

                    # Save LBP output
                    lbp_img = np.multiply(lbp(img), 32)
                    cv2.imwrite(os.path.join(lbp_folder, f"{img_counter}.png"), lbp_img)

                    # Save masked output
                    masked = masking(lbp_img, binary_map)
                    cv2.imwrite(os.path.join(masked_folder, f"{img_counter}.png"), masked)
                
                    feature_appender(masked)
                    feature_appender(lbp_img)
                    feature_appender(img[0])
                    feature_appender(img[1])
                    feature_appender(img[2])
                    
                    # Transpose summed_features to make it a row vector
                    row_features = np.reshape(row_vector, (1, -1))
                    X.extend(row_features)
                    
                    # Reset row_vector for each image
                    row_vector = np.array([])
                    
                    y.append(class_label)
                    
                    img_counter += 1
        
        return X, y

    def run(self):
        X_train, y_train = self.create_data(self.train_folder)
        X_test, y_test = self.create_data(self.test_folder)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)
        print("...imputing")
        
        
        # Save training features to a CSV file
        train_features_df = pd.DataFrame(X_train)
        train_features_df["label"] = y_train
        train_features_df.to_csv(self.train_features_csv, index=False)
        print("...training")

        # Train and test SVM
        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print("...predictions")

        # Calculate prediction statistics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=self.classes).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

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
svm.run()
