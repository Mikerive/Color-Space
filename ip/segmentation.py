import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageutils as iu


def segment_image(img):
    # Calculate the histogram of the image
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 255))

    # Find the peaks in the histogram using the find_peaks function from SciPy
    from scipy.signal import find_peaks
    
    # Set minimum peak height to 1/10 the maximum height of the image
    peaks, _ = find_peaks(hist, height=np.max(hist) // 10)

    # Find the ranges of pixel values corresponding to halfway between each peak
    ranges = []
    for i in range(len(peaks)):
        if i == 0:
            start = 0
        else:
            start = (peaks[i-1] + peaks[i]) // 2
        
        # If not the last peak, the next center is halfway between the two peaks, else it continues till the end.
        end = (peaks[i] + peaks[i+1]) // 2 if i < len(peaks) - 1 else 255
        ranges.append((start, end))

    # Create a binary mask for each range of pixel values
    masks = []
    for r in ranges:
        # Values within the range arr assigned 255, the rest 0
        mask = np.zeros_like(img)
        mask[(img >= r[0]) & (img <= r[1])] = 255
        masks.append(mask)

    # Show the original image and its histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax2.bar(bins[:-1], hist, width=1)
    ax2.set_title('Histogram') 
    plt.show()

    # Show the identified objects
    iu.plot_images_with_histograms(masks)

    return masks



def global_threshold(image, mode = 'mean', deltaT = 3):
    """
    Applies global thresholding to an input grayscale image.
    
    :param image: A grayscale image as a NumPy array.
    :param threshold: The threshold value.
    :return: The thresholded image as a binary NumPy array.
    """
    image = image.astype(int)
    threshold = None
    
    if mode == 'mean':
        threshold = (np.max(image)+np.min(image))//2
    elif mode == 'median':
        threshold = np.median(image)
        
    # Minimizes the intra-class variance of the two resulting classes (foreground and background)
    elif mode == 'otsu':
        _, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return output
    
    # If T is within three pixels of the last T, update and terminate while loop
    done = False
    while done == False:
        img1, img2 = np.zeros_like(image)
        img1, img2 = image[image<threshold], image[image>threshold]
        thresholdnext = (np.mean(img1)+np.mean(img2))//2
        if abs(thresholdnext-threshold) < deltaT:
            done = True
        threshold = thresholdnext
    
    
    # Create a binary mask by comparing the image with the threshold value
    mask = np.zeros_like(image)
    mask[image >= threshold] = 1

    # Multiply the mask by 255 to obtain a binary image with 0s and 255s
    output = mask * 255

    return output


def global_multiple_threshold(image, thresholds, mode = 'mean'):
    
    if mode == 'mean':
        threshold = (np.max(image)+np.min(image))/2
    
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the segmented image to all black pixels
    segmented = np.zeros_like(image)

    # For each threshold, create a binary image and merge it with the segmented image
    for threshold in thresholds:
        binary = np.where(image >= threshold, 255, 0)
        segmented = np.maximum(segmented, binary)

    # Apply a median filter to reduce noise
    segmented = cv2.medianBlur(segmented, 3)

    return segmented
