import numpy as np


class Kernels():
    def __init__(self):
        pass
    def prewitt_vertical_kernel(self, size):
        """
        Generates the Vertical Prewitt Kernel of specified size
        """
        # Create the Prewitt matrix.
        kernel = np.zeros((size, size))

        for j in range(size):
            if j < size // 2:
                kernel[:, j] = -1
            elif j > size // 2:
                kernel[:, j] = 1

        return kernel

    def gaussian_kernel(n, sigma):
        x, y = np.meshgrid(np.arange(-n // 2 + 1, n // 2 + 1),
                           np.arange(-n // 2 + 1, n // 2 + 1))
        g = np.exp(-(x**2 + y**2) // (2 * sigma**2))
        return g

    def LoG_kernel(n, sigma):

        # Create a 1-D array of indices, centering them around 0
        ind = np.arange(-n // 2, n // 2 + 1)

        # Create 2D arrays of zeros for the LoG filter and Gaussian filter
        L, G = np.meshgrid(ind, ind, indexing='ij')
        LoG_filter = (-1 / (np.pi * sigma ** 4)) * (1 - (L ** 2 + G ** 2) /
                                                    (2 * sigma ** 2)) * np.exp(-(L ** 2 + G ** 2) / (2 * sigma ** 2))

        # Normalize the filter so that its values sum up to 0
        return LoG_filter / np.sum(LoG_filter)