import cv2
import numpy as np

import os
import cv2
import numpy as np

# Define function to extract HOG features from an image
def extract_hog(image):
    win_size = (256, 256)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_desc = hog.compute(image)
    return hog_desc.flatten()

# Create lists for positive and negative HOG descriptors
hog_desc_pos = []
hog_desc_neg = []

# Load positive samples
pos_dir = "positive_samples"
for filename in os.listdir(pos_dir):
    img_path = os.path.join(pos_dir, filename)
    img = cv2.imread(img_path)
    hog_desc = extract_hog(img)
    hog_desc_pos.append(hog_desc)

# Load negative samples
neg_dir = "negative_samples"
for filename in os.listdir(neg_dir):
    img_path = os.path.join(neg_dir, filename)
    img = cv2.imread(img_path)
    hog_desc = extract_hog(img)
    hog_desc_neg.append(hog_desc)

# Stack negative hog and negative hog into hog_desc_samples
hog_desc_samples = np.vstack((hog_desc_pos, hog_desc_neg))
labels = np.hstack((np.ones(len(hog_desc_pos)), np.zeros(len(hog_desc_neg))))

# Train an SVM classifier on a set of positive and negative samples (images)
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(hog_desc_samples, cv2.ml.ROW_SAMPLE, labels)

# set the path to your test images folder
test_folder = 'test_images'

def object_identification(img_tgt, img_label, threshold, svm):
    # Compute the HOG descriptors for both images
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_desc_tgt = hog.compute(img_tgt)

    # Compute the HOG descriptor for the target image
    scores = []
    for y in range(0, img_tgt.shape[0] - win_size[1], win_size[1] // 2):
        for x in range(0, img_tgt.shape[1] - win_size[0], win_size[0] // 2):
            hog_desc_win = hog_desc_tgt[y:y+win_size[1], x:x+win_size[0]]
            score = svm.predict(hog_desc_win.reshape(1, -1))[1][0][0]
            scores.append((x, y, score))

    # Apply non-maximum suppression on the heatmap
    nms_threshold = 0.3
    detections = []
    while len(scores) > 0:
        best = max(scores, key=lambda x: x[2])
        detections.append(best)
        scores.remove(best)
        for other in scores:
            if cv2.intersection_over_union(best, other) > nms_threshold:
                scores.remove(other)

    # Draw bounding boxes around the final detections
    for detection in detections:
        x, y, score = detection
        if score >= threshold:
            cv2.rectangle(img_tgt, (x, y), (x + win_size[0], y + win_size[1]), (0, 255, 0), 2)

    # Show the result image
    cv2.imshow(f'{img_label}', img_tgt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# loop through each image file in the folder
for file_name in os.listdir(test_folder):
    # check if the file is an image file
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # load the image
        img_tgt = cv2.imread(os.path.join(test_folder, file_name))
        
        # call the object_identification function with appropriate parameters
        object_identification(img_tgt, 30, svm, file_name)