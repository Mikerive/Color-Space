from ImageLab import *
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count


orange1 = Image.open('Image/orange1.jpg').convert('RGB')
orange1 = np.asarray(orange1)
orange1 = ColorSpace(orange1).rgb_to_grayscale()

name = 'orange1'
horizontal_sharpening = np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]])
vertical_sharpening = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]])
ordered_matrix = np.array([[0, 1, 0], [1, 3, 1], [0, 1, 0]])
alpha_matrix = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
orange = np.expand_dims(orange1, axis=2)

orange = Convolution(orange, f'amean_{name}').sliding_kernel(vertical_sharpening, Convolution.weighted_arithmetic_mean)
