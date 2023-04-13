import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List

def rht_lines(image: np.ndarray, threshold: int, num_iterations: int) -> List[tuple]:
    rows, cols = image.shape
    accumulator = np.zeros((rows, cols), dtype=np.uint8)

    edges = np.argwhere(image > 0)
    for _ in range(num_iterations):
        # Randomly pick two points in the edge image
        y1, x1 = random.choice(edges)
        y2, x2 = random.choice(edges)

        # Skip iteration if points are the same
        if x1 == x2 and y1 == y2:
            continue

        # Calculate line parameters
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Increment the accumulator for the line
        ys = np.arange(rows)
        xs = np.array((ys - b) / m, dtype=np.int32)
        valid_xs = xs[(xs >= 0) & (xs < cols)]
        accumulator[ys[(xs >= 0) & (xs < cols)], valid_xs] += 1

    lines = list(zip(*np.where(accumulator > threshold)))

    return lines

def find_contours(img):

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    output = np.zeros_like(img)
    # Draw contours on the original image
    return cv2.drawContours(output, contours, -1, (0, 255, 0), 3)


# # Takes GreyScale image, calculates the most common line using Randomized Hough Transform
# def rht_lines(image, threshold, num_iterations):

#     rows, cols = image.shape
#     accumulator = np.zeros((rows, cols), dtype=np.uint8)

#     for _ in range(num_iterations):
#         # Randomly pick two points in the edge image
#         y1, x1 = random.choice(np.argwhere(image > 0))
#         y2, x2 = random.choice(np.argwhere(image > 0))

#         # Skip iteration if points are the same
#         if x1 == x2 and y1 == y2:
#             continue

#         # Calculate line parameters
#         m = (y2 - y1) / (x2 - x1)
#         b = y1 - m * x1

#         # Increment the accumulator for the line
#         for x in range(cols):
#             y = int(m * x + b)
#             if 0 <= y < rows:
#                 accumulator[y, x] += 1

#     lines = []
#     # Extract lines with enough votes
#     for y in range(rows):
#         for x in range(cols):
#             if accumulator[y, x] > threshold:
#                 lines.append((y, x))

#     return lines


def draw_lines(image, lines):
    for y, x in lines:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    input_image = cv2.imread('X:/Programs/Research_Project/T2.jpg')
    threshold = 30
    num_iterations = 1000

    lines = rht_lines(input_image, threshold, num_iterations)
    output_image = draw_lines(input_image, lines)

    cv2.imshow('Output Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
