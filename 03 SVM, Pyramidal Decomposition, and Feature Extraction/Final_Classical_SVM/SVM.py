import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.util import view_as_windows
from scipy.stats import skew, kurtosis
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import os
import cv2
import matplotlib.pyplot as plt
import sys

# from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QMainWindow
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
# from PyQt5.QtCore import Qt, QPoint

class SupportVectorMachine:
    def __init__(self, window_size=21):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.clf = svm.SVC(kernel='linear')

    def extract_features(self, img_path):
        
        img = cv2.imread(img_path)
        
        # Apply Local Binary Patterns to bgr img and extract windows
        lbp_radius = 3
        lbp_n_points = 8 * lbp_radius
        
        b = local_binary_pattern(img[:,:,0], lbp_n_points, lbp_radius, 'uniform')
        g = local_binary_pattern(img[:,:,1], lbp_n_points, lbp_radius, 'uniform')
        r = local_binary_pattern(img[:,:,2], lbp_n_points, lbp_radius, 'uniform')
        bgr_img = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

        # x, y, window_x, window_y, rgb
        bgr_windows = self.get_windows(bgr_img)
        
        
        
        # Apply Local Binary Patterns to gray img and extract windows
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and direction
        magnitude = cv2.magnitude(sobelx, sobely)
        # direction = cv2.phase(sobelx, sobely, angleInDegrees=True)
        
        gray_lbp = local_binary_pattern(magnitude, lbp_n_points, lbp_radius, 'uniform')
        
        # x, y, window_x, window_y
        gray_windows = self.get_windows(gray_lbp)
        
        print(gray_windows.shape)
        
        # hog_features = np.apply_along_axis(hog, 0, gray_lbp, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2-Hys')

        # Calculate color histograms
        num_bins = 11
        gray_hist = np.array([np.histogram(gray_window, bins=num_bins)[0] for gray_window in gray_windows])
        red_hist = np.array([np.histogram(window[:,:,2], bins=num_bins, range=(0, 1))[0] for window in bgr_windows])
        green_hist = np.array([np.histogram(window[:,:,1], bins=num_bins, range=(0, 1))[0] for window in bgr_windows])
        blue_hist = np.array([np.histogram(window[:,:,0], bins=num_bins, range=(0, 1))[0] for window in bgr_windows])
        
        print(red_hist.shape)
        
        # Compute mean, median, variance, skewness, and kurtosis for each color channel histogram
        color_features = []
        for hist in [red_hist, green_hist, blue_hist, gray_hist]:
            hist_mean = np.mean(hist, axis=1)
            hist_median = np.median(hist, axis=1)
            hist_variance = np.var(hist, axis=1)
            hist_skewness = skew(hist, axis=1)
            hist_kurtosis = kurtosis(hist, axis=1)
            color_features.append(np.column_stack((hist_mean, hist_median, hist_variance, hist_skewness, hist_kurtosis)))

        # Concatenate color features
        color_features_combined = np.concatenate(color_features, axis=1)
        
        # Impute missing values in the color_features_combined array
        imputer = SimpleImputer(strategy='mean')
        imputed_features = imputer.fit_transform(color_features_combined)

        return self.scaler.fit_transform(imputed_features)

    def train(self, img_path, labels):
        X = self.extract_features(img_path)
        print(X.shape)
        print(labels.shape)
        self.clf.fit(X, labels)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        X = self.extract_features(img_path)
        predictions = self.clf.predict(X)
        return predictions.reshape(img.shape[0], img.shape[1])

    def get_windows(self, img):
        
        pad_size = self.window_size // 2
        padded_image = np.pad(img, pad_width=pad_size, mode='reflect')
        
        if img.ndim ==3 and img.shape[-1] == 3:  # Check if the image is an RGB image (3D array)
            h, w, c = padded_image.shape
            windows = view_as_windows(padded_image, (self.window_size, self.window_size, c))
            windows = windows.reshape(-1, self.window_size, self.window_size, c)
        else:  # Handle grayscale images as before (2D array)
            windows = view_as_windows(padded_image, (self.window_size, self.window_size))
            windows = windows.reshape(-1, self.window_size, self.window_size)
        return windows

class Labeler:
    def __init__(self):
        self.draw_labeling = False
        self.last_x = None
        self.last_y = None
        self.img = None
        self.labels = None
        self.display_img = None
    
    def draw_line(self, x, y):
        cv2.line(self.labels, (self.last_x, self.last_y), (x, y), 1, thickness=20)
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
        return self.labels.flatten()


def overlay_labels_on_img(img_path, labels):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    labels = labels.reshape(img.shape[:,:,0])
    overlay_img = cv2.addWeighted(img, 0.7, labels * 255, 0.3, 0)
    return overlay_img

features_list = []
labels = []

input_image_path = "C:/Programs/Image Processing/Color Space/FeatureExtraction/test_images/T9.jpg"
labeler = Labeler()
labels = labeler.get_labels(input_image_path)

# Initialize an SVM classifier and train it with the labeled image
svm_classifier = SupportVectorMachine(window_size=21)
svm_classifier.train(input_image_path, labels)
labels = svm_classifier.predict(input_image_path)

overlay_img = overlay_labels_on_img(input_image_path, labels)

cv2.imshow('Prediction', overlay_img)



        
# # Load positive images (class 1)
# for filename in os.listdir(positive_image_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Load the image
#         img = cv2.imread(os.path.join(positive_image_dir, filename))

#         # Extract features from the image
#         features = svm_classifier.extract_features(img)

#         # Add features and label (class 1) to lists
#         features_list.append(features)
#         labels_list.append(1)

# # Load negative images (class 0)
# for filename in os.listdir(negative_image_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # Load the image
#         img = cv2.imread(os.path.join(negative_image_dir, filename))

#         # Extract features from the image
#         features = svm_classifier.extract_features(img)

#         # Add features and label (class 0) to lists
#         features_list.append(features)
#         labels_list.append(0)

# # Convert lists to numpy arrays
# X = np.array(features_list)
# y = np.array(labels_list)


















# import sys
# from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QMainWindow
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
# from PyQt5.QtCore import Qt, QPoint
# import cv2
# import numpy as np

# class InteractiveLabeling(QMainWindow):
#     def __init__(self, img):
#         super().__init__()

#         self.img = img
#         self.labels = np.zeros_like(img, dtype=np.uint8)

#         self.setWindowTitle('Interactive Labeling')

#         qimage_data = np.ndarray(shape=img.shape[::-1], dtype=np.uint8, buffer=img.data)
#         self.image = QImage(qimage_data.data, img.shape[1], img.shape[0], QImage.Format_Grayscale8)

#         self.scene = QGraphicsScene(self)
#         self.view = QGraphicsView(self.scene, self)
#         self.view.setRenderHint(QPainter.Antialiasing)
#         self.view.setRenderHint(QPainter.SmoothPixmapTransform)

#         self.pixmap = QPixmap.fromImage(self.image)
#         self.scene.addPixmap(self.pixmap)

#         self.view.setSceneRect(0, 0, self.pixmap.width(), self.pixmap.height())
#         self.setCentralWidget(self.view)

#         self.drawing = False
#         self.last_point = QPoint()

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = True
#             self.last_point = self.view.mapToScene(event.pos()).toPoint()

#     def mouseMoveEvent(self, event):
#         if self.drawing:
#             current_point = self.view.mapToScene(event.pos()).toPoint()

#             painter = QPainter(self.pixmap)
#             painter.setPen(QPen(QColor(255, 255, 255), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#             painter.drawLine(self.last_point, current_point)
#             self.pixmap_item.setPixmap(self.pixmap)
#             self.last_point = current_point

#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.drawing = False

#     def get_labels(self):
#         self.labels = np.array(self.pixmap.toImage().bits()).reshape(self.pixmap.height(), self.pixmap.width(), self.pixmap.depth() // 8)

#         return self.labels.ravel()

# def get_labels(input_image_path):
#     img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

#     app = QApplication(sys.argv)
#     interactive_labeling = InteractiveLabeling(img)
#     interactive_labeling.show()
#     app.exec_()

#     labels = interactive_labeling.get_labels()
#     return labels
