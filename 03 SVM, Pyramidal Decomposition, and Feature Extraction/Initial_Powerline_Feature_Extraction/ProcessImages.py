import os
import numpy as np
import cv2

from ImageLab import *

# set the path to your test images folder
test_folder = 'TrainImages'

images = []
labels = []

# loop through each image file in the folder
for file_name in os.listdir(test_folder):
    # check if the file is an image file
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        
        img_path = os.path.join(test_folder, file_name)
        file_name = os.path.splitext(file_name)[0]
        _,_ = ColorSpace(img_path, test_folder, file_name).process(rgb_to_grayscale())
        _,inverted_path = Filters(img_path, test_folder, file_name).process(inversion())
        dilated, dilated_path = ImageProcessor(inverted_path, 'intermediate', 'd').process(Dilation(), np.full((5, 5), 1))
        eroded, eroded_path = ImageProcessor(inverted_path, 'intermediate', 'e').process(Erosion(), np.full((5, 5), 1))
        
        diff, diff_path = MultiProcessor(dilated_path, eroded_path, test_folder, file_name).process(Difference())
        gamma, gamma_path = Filters(diff_path, test_folder, file_name).process(gamma_correction(1.5))
        segmented, segmented_path = Segment(gamma_path, test_folder, file_name).process(Adaptive_Global_Threshold())
        cv2.imwrite(img_path, segmented)
        
        
