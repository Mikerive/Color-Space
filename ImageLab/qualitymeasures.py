import numpy as np



class QualityMeasures:
    def __init__(self, img):
        self.img = np.array(img)
    
    # EME - we want to maximize the difference between I max and I min. C should be 1.
    def measure_EME(self, img):
        # Compute the amplitude of the color range
        amplitude = np.ptp(img) / 2

        # Compute the EME
        EME = amplitude ** 2

        return EME
    
    def histogram_distance_euclidian(self, other_img : np.ndarray):
        
        # For each layer, calculate, the average absolute difference. Divide the sum for each layer by
        # the number of pixels in the image. Then average the differences for each layer.
        
        # Calculate the image histogram
        histogram1, bin_edges = np.histogram(
            self.img.flatten(), bins=256, range=(0, 256))
        # Calculate the image histogram
        histogram2, bin_edges = np.histogram(
            other_img.flatten(), bins=256, range=(0, 256))
        return np.sum(np.abs(histogram1 - histogram2))