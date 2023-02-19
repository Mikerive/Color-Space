import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.show()


def show_histograms(histograms, titles):
    # Determine the number of histograms to display
    num_histograms = len(histograms)

    # Determine the number of rows and columns for the subplot grid
    num_rows = int(np.ceil(num_histograms / 2))
    num_cols = 2 if num_histograms >= 2 else 1

    # Create a new figure with a subplot grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 6))

    # Loop through the histograms and titles and plot each one
    for i, (hist, title) in enumerate(zip(histograms, titles)):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.bar(np.arange(len(hist)), hist, width=1)
        ax.set_xlim(0, len(hist))
        ax.set_title(title)

    # Remove any unused subplots and display the figure
    fig.tight_layout()
    plt.show()
