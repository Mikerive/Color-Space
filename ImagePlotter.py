import matplotlib.pyplot as plt
import numpy as np
import cv2

class ImagePlotter:
    def __init__(self, img):
        self.img = img
    
    def plot_histogram_cdf(self):
        # Compute the histogram and CDF of the image
        hist, bins = np.histogram(self.img.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Plot the histogram and CDF
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
        ax[0].hist(self.img.flatten(), 256, [0, 256], color='r')
        ax[0].set_xlim([0, 256])
        ax[0].set_ylim([0, self.img.size])
        ax[0].set_title('Histogram')
        ax[0].set_ylabel('Frequency')
        ax[1].plot(cdf_normalized, color='b')
        ax[1].set_xlim([0, 256])
        ax[1].set_ylim([0, 1])
        ax[1].set_title('CDF')
        ax[1].set_xlabel('Pixel value')
        ax[1].set_ylabel('Normalized frequency')
        plt.show()

    def plot_HSV(self):
        # Convert the RGB image to HSV color space
        R, G, B = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]
        V = np.max(self.img, axis=2)
        S = (V - np.min(self.img, axis=2)) / V
        H = np.zeros_like(V)
        H[np.where(V == R)] = 60 * ((G[np.where(V == R)] - B[np.where(V == R)]) /
                                    (V[np.where(V == R)] - np.min(self.img, axis=2)[np.where(V == R)])) % 360
        H[np.where(V == G)] = 60 * (2 + (B[np.where(V == G)] - R[np.where(V == G)]) /
                                    (V[np.where(V == G)] - np.min(self.img, axis=2)[np.where(V == G)])) % 360
        H[np.where(V == B)] = 60 * (4 + (R[np.where(V == B)] - G[np.where(V == B)]) /
                                    (V[np.where(V == B)] - np.min(self.img, axis=2)[np.where(V == B)])) % 360

        # Plot the histograms of H, S, and V channels
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax[0, 0].imshow(H, cmap='hsv', vmin=0, vmax=180)
        ax[0, 0].set_title('Hue')
        ax[0, 1].imshow(S, cmap='gray', vmin=0, vmax=1)
        ax[0, 1].set_title('Saturation')
        ax[0, 2].imshow(V, cmap='gray', vmin=0, vmax=255)
        ax[0, 2].set_title('Value')
        for i, channel in enumerate([H, S, V]):
            ax[1, i].hist(channel.flatten(), bins=256, range=(
                0, 255), color=['red', 'green', 'blue'][i])
            ax[1, i].set_xlim([0, 255])
            ax[1, i].set_title(['Hue', 'Saturation', 'Value'][i] + ' Histogram')
            ax[1, i].set_xlabel('Pixel Value')
            ax[1, i].set_ylabel('Frequency')
        plt.show()

    def plot_RGB(self):
         # Separate the red, green, and blue channels
        r_channel = self.img[:, :, 2]
        g_channel = self.img[:, :, 1]
        b_channel = self.img[:, :, 0]

        # Plot the red, green, and blue channels and their histograms
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 1].imshow(r_channel, cmap='gray')
        axes[0, 1].set_title('Red channel')
        axes[0, 2].hist(r_channel.flatten(), bins=256, range=(0, 256), color='r')
        axes[0, 2].set_title('Red channel histogram')
        axes[1, 1].imshow(g_channel, cmap='gray')
        axes[1, 1].set_title('Green channel')
        axes[1, 2].hist(g_channel.flatten(), bins=256, range=(0, 256), color='g')
        axes[1, 2].set_title('Green channel histogram')
        axes[1, 0].imshow(b_channel, cmap='gray')
        axes[1, 0].set_title('Blue channel')
        axes[1, 2].hist(b_channel.flatten(), bins=256, range=(0, 256), color='b')
        axes[1, 2].set_title('Blue channel histogram')
        plt.tight_layout()
        plt.show()