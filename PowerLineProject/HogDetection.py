import os
import cv2
import argparse
from sklearn.svm import LinearSVC
from skimage import feature
from skimage.filters import threshold_otsu

import numpy as np

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description', 
                    choices=['FarTowers', 'LargeTowers', 'MediumTowers'])
args = vars(parser.parse_args())

images = []
labels = []
# get all the image folder paths


image_dir = f"Towers/{args['path']}"  # get the directory path
image_files = os.listdir(image_dir)  # get the list of image files

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)  # create the full image path by concatenating the dir path with the image filename
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (250, 250))

    # get the HOG descriptor for the image
    hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

    # update the data and labels
    images.append(hog_desc)
    labels.append(image_file) # We will append only the filename here

print(labels)
   
# train Linear SVC 
print('Training on train images...')    
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)


        
# predict on the test images
print('Evaluating on test images...')
# loop over the test dataset folders
for (i, imagePath) in enumerate(os.listdir(f"test_images")):
    image = cv2.imread(f"test_images/{imagePath}", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (250, 250))
    # get the HOG descriptor for the test image
    (hog_desc, hog_image) = feature.hog(resized_image, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    # prediction
    pred = svm_model.predict(hog_desc.reshape(1, -1))[0]
    # convert the HOG image to appropriate data type. We do...
    # ... this instead of rescaling the pixels from 0. to 255.
    hog_image = hog_image.astype('float64')


    # show thw HOG image
    cv2.imshow('HOG Image', hog_image)
    # put the predicted text on the test image
    cv2.putText(image, pred.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 255, 0), 2)
    cv2.imshow('Test Image', image)
    cv2.imwrite(f"outputs/{args['path']}_hog_{i}.jpg", hog_image*255.) # multiply by 255. to bring to OpenCV pixel range
    cv2.imwrite(f"outputs/{args['path']}_pred_{i}.jpg", image)
    cv2.waitKey(0)
    
    
    
    
    
    #  # keep a minimum image size for accurate predictions
# if image.shape[1] < 400: # if image width < 400
#     (height, width) = image.shape[:2]
#     ratio = width / float(width) # find the width to height ratio
#     image = cv2.resize(image, (400, width*ratio)) # resize the image according to the width to height ratio
    
    
    
    # # perform object detection by thresholding the HOG image
    # thresh = threshold_otsu(hog_image)
    # binary = hog_image > thresh
    
    # # find contours in the binary image
    # contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # draw rectangles around the detected objects
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w*h < 50: # ignore small objects
    #         continue
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)    