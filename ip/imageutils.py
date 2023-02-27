import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image

__all__ = ["plot_HSV", "plot_RGB", "plot_image_with_histogram", "save_image_to_folder",
           "load_image_from_file", "create_subimage", "get_image_type", "plot_images_with_histograms",
           "plot_images_and_noise_with_histograms", "split_image",
           "split_image_nxn_sections", "merge_sections_into_image", "apply_function_nxn_sections",
           "show_image_sections"]


def plot_HSV(img):
    
    # Separate the layers 0, 1, and 2
    H = img[:, :, 0]
    S = img[:, :, 1]
    V = img[:, :, 2]
    
    # Plot the red, green, and blue channels and their histograms
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    
    # Set the spacing between the subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    rgb_img = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_HSV2RGB)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original')

    axes[1, 0].hist(img.flatten(), bins=256,
                    range=(0, 256), color='gray')
    axes[1, 0].set_title('Original histogram')

    axes[0, 1].imshow(H, cmap='hsv')
    axes[0, 1].set_title('Hue')

    axes[1, 1].hist(H.flatten(), bins=256,
                    range=(0, 256), color='gray')
    axes[1, 1].set_title('Hue histogram')

    axes[0, 2].imshow(S, cmap='Greys')
    axes[0, 2].set_title('Saturation channel')

    axes[1, 2].hist(S.flatten(), bins=256,
                    range=(0, 256), color='gray')
    axes[1, 2].set_title('Saturation histogram')

    axes[0, 3].imshow(V, cmap='Greys')
    axes[0, 3].set_title('Value channel')

    axes[1, 3].hist(V.flatten(), bins=256,
                    range=(0, 256), color='gray')
    axes[1, 3].set_title('Value histogram')

def plot_RGB(img):
    # Separate the red, green, and blue channels
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 0]

    # Plot the red, green, and blue channels and their histograms
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

    # Set the spacing between the subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.4)

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')

    axes[1, 0].hist(img.flatten(), bins=256,
                    range=(0, 256), color='gray')
    axes[1, 0].set_title('Original histogram')

    axes[0, 1].imshow(r_channel, cmap='Reds')
    axes[0, 1].set_title('Red channel')

    axes[1, 1].hist(r_channel.flatten(), bins=256,
                    range=(0, 256), color='red')
    axes[1, 1].set_title('Red channel histogram')

    axes[0, 2].imshow(g_channel, cmap='Greens')
    axes[0, 2].set_title('Green channel')

    axes[1, 2].hist(g_channel.flatten(), bins=256,
                    range=(0, 256), color='green')
    axes[1, 2].set_title('Green channel histogram')

    axes[0, 3].imshow(b_channel, cmap='Blues')
    axes[0, 3].set_title('Blue channel')

    axes[1, 3].hist(b_channel.flatten(), bins=256,
                    range=(0, 256), color='blue')
    axes[1, 3].set_title('Blue channel histogram')

    plt.tight_layout()
    plt.show()
    
def plot_image_with_histogram(img, title):
    
    # Calculate the image histogram
    histogram, bin_edges = np.histogram(img.flatten(), bins=256, range=(0, 256))

    # Create a new figure with two subplots: one for the image and one for the histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    type = get_image_type(img)

    # Plot the image in the first subplot
    ax1.imshow(img)
    ax1.set_title(title)

    # Plot the histogram in the second subplot
    ax2.bar(bin_edges[:-1], histogram, width=1)
    ax2.set_xlim(left=0, right=256)
    ax2.set_title('{} Histogram'.format(title))
    
    # Calculate the midpoint value, mean, and variance of the histogram
    midpoint = (np.argmax(hist) + 1) / 2
    mean = np.mean(img)
    variance = np.var(img)

    # Add the midpoint value, mean, and variance to the plot
    ax2.axvline(x=midpoint, color='r', linestyle='--',
                       label=f"Midpoint: {midpoint:.2f}")
    ax2.axvline(x=mean, color='g', linestyle='--',
                       label=f"Mean: {mean:.2f}")
    ax2.axvline(x=mean-np.sqrt(variance), color='b',
                       linestyle='--', label=f"Variance: {variance:.2f}")
    ax2.axvline(x=mean+np.sqrt(variance), color='b', linestyle='--')
    ax2.legend()

    # Display the image and histogram
    plt.show()

def save_image_to_folder(img, folder_name, filename):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Convert image to the appropriate color space if necessary
    if len(img.shape) == 2:
        mode = 'L'
    elif img.shape[2] == 3:
        mode = 'RGB'
    elif img.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError('Unsupported image format')
    if mode == 'RGB':
        pil_img = Image.fromarray(img)
    elif mode == 'L':
        pil_img = Image.fromarray(img, mode=mode)
    elif mode == 'RGBA':
        pil_img = Image.fromarray(img, mode=mode)
        pil_img = pil_img.convert('RGB').convert('RGBA')

    # Save the image to the new folder under the same name
    new_path = os.path.join(folder_name, filename)
    pil_img.save(new_path)

    return new_path

def load_image_from_file(filepath):
    # Load the image from the file
    pil_img = Image.open(filepath)

    # Convert image to NumPy array in the appropriate format
    if pil_img.mode == 'L':
        img = np.array(pil_img)
    elif pil_img.mode == 'RGB':
        img = np.array(pil_img)
    elif pil_img.mode == 'RGBA':
        img = np.array(pil_img.convert('RGB'))
        alpha = np.array(pil_img)[:, :, 3]
        img = np.dstack((img, alpha))

    return img

def create_subimage(img, x_start, y_start, x_end, y_end):
    # Create the subimage
    subimg = img[y_start:y_end, x_start:x_end, :]

    return subimg

def get_image_type(image):
    """
    Returns the color type of an image as a string: 'RGB', 'HSV', or 'Grayscale'.
    """
    if len(image.shape) == 2:
        return 'Grayscale'
    elif image.shape[2] == 3 and cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() > 10:
        return 'RGB'
    else:
        return 'HSV'
    
def plot_images_with_histograms(images, titles = None):
    """
    Plots an array of images vertically with histograms for each image.
    """
    fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(8, 3*len(images)))
    plt.subplots_adjust(hspace=0.3)

    for i, img in enumerate(images):
        
        type = get_image_type(img)
        
        if type == 'Grayscale':
            axes[i, 0].imshow(img, cmap='gray')
        if type == 'RGB':
            axes[i, 0].imshow(img)
        if type == 'HSV':    
            axes[i, 0].imshow(img, cmap='hsv')    
        # Plot image
        
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        if titles == None:
            axes[i, 0].set_title("Image {}".format(i+1))
        else:
            axes[i, 0].set_title("{}".format(titles[i]))

        # Plot histogram
        hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
        axes[i, 1].plot(bins[:-1], hist, lw=2)
        axes[i, 1].set_xlim([0, 256])
        axes[i, 1].set_ylim([0, np.max(hist)+100])
        axes[i, 1].set_title("Histogram")
    
        # Calculate the midpoint value, mean, and variance of the histogram
        midpoint = (np.argmax(hist) + 1) / 2
        mean = np.mean(img)
        variance = np.var(img)

        # Add the midpoint value, mean, and variance to the plot
        axes[i, 1].axvline(x=midpoint, color='r', linestyle='--',
                        label=f"Midpoint: {midpoint:.2f}")
        axes[i, 1].axvline(x=mean, color='g', linestyle='--', label=f"Mean: {mean:.2f}")
        axes[i, 1].axvline(x=mean-np.sqrt(variance), color='b',
                        linestyle='--', label=f"Variance: {variance:.2f}")
        axes[i, 1].axvline(x=mean+np.sqrt(variance), color='b', linestyle='--')
        axes[i, 1].legend()

    plt.show()
    
def plot_images_and_noise_with_histograms(images, noises, titles=None):
    """
    Plots an array of images vertically with histograms for each image.
    """
    
    fig, axes = plt.subplots(
        nrows=len(images), ncols=3, figsize=(8, 3*len(images)))
    plt.subplots_adjust(hspace=0.3)

    for i, img in enumerate(images):
        # Plot image
        type = get_image_type(img)

        if type == 'Grayscale':
            axes[i, 0].imshow(img, cmap='gray')
        if type == 'RGB':
            axes[i, 0].imshow(img)
        if type == 'HSV':
            axes[i, 0].imshow(img, cmap='hsv')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if titles == None:
            axes[i, 0].set_title("Image {}".format(i+1))
        else:
            axes[i, 0].set_title("{}".format(titles[i]))
        
        # Plot Noise    
        axes[i, 1].imshow(noises[i], cmap='gray')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        if titles == None:
            axes[i, 1].set_title("Noise {}".format(i+1))
        else:
            axes[i, 1].set_title("{} Noise".format(titles[i]))

        # Plot histogram
        hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
        axes[i, 2].plot(bins[:-1], hist, lw=2)
        axes[i, 2].set_xlim([0, 256])
        axes[i, 2].set_ylim([0, np.max(hist)+100])
        if titles == None:
            axes[i, 2].set_title("Histogram {}".format(i+1))
        else:
            axes[i, 2].set_title("{}".format(titles[i]))
            
        # Calculate the midpoint value, mean, and variance of the histogram
        midpoint = (np.argmax(hist) + 1) / 2
        mean = np.mean(img)
        variance = np.var(img)

        # Add the midpoint value, mean, and variance to the plot
        axes[i, 2].axvline(x=midpoint, color='r', linestyle='--',
                           label=f"Midpoint: {midpoint:.2f}")
        axes[i, 2].axvline(x=mean, color='g', linestyle='--',
                           label=f"Mean: {mean:.2f}")
        axes[i, 2].axvline(x=mean-np.sqrt(variance), color='b',
                           linestyle='--', label=f"Variance: {variance:.2f}")
        axes[i, 2].axvline(x=mean+np.sqrt(variance), color='b', linestyle='--')
        axes[i, 2].legend()
        
    plt.show()


# Functions to apply a function to sections

def split_image_nxn_sections(img, sections):
    
    # Get the size of the image
    height, width = img.shape[:2]

    # Calculate the height of each section
    section_height = int(np.ceil(height / sections))

    # Calculate the width of each section
    section_width = int(np.ceil(width / sections))

    # Initialize the list to store the sections
    section_list = []

    # Split the image into sections
    for row in range(0, height, section_height):
        for col in range(0, width, section_width):
            section = img[row:row + section_height, col:col + section_width]
            section_list.append(section)

    # Return the output wrapped in a dictionary
    return {
        'section_list': section_list,
        'section_height': section_height,
        'section_width': section_width,
        'height': height,
        'width': width,
    }

def merge_sections_into_image(section_dict):
    # Get the number of channels from the first section
    num_channels = section_dict['section_list'][0].shape[2]

    # Initialize the result image with the correct number of channels
    result_img = np.zeros((section_dict['height'], section_dict['width'], num_channels), dtype=np.uint8)

    # Merge the sections into a single image
    index = 0
    for row in range(0, section_dict['height'], section_dict['section_height']):
        for col in range(0, section_dict['width'], section_dict['section_width']):
            section = section_dict['section_list'][index]
            result_img[row:row + section_dict['section_height'],
                       col:col + section_dict['section_width'], :] = section
            index += 1

    return result_img

def func_pass(x): return x

def apply_function_nxn_sections(section_dict, func1=[func_pass], func2=[func_pass], func3=[func_pass], *args):
    
    L, A, B = cv2.split(section_dict['section_list'])

    L = [func1(section[:,:,0], *args) for section in section_dict['section_list']]
    A = [func2(section[:,:,1], *args) for section in section_dict['section_list']]
    B = [func3(section[:,:,2], *args) for section in section_dict['section_list']]
    
    # Apply the functions to each section
    results = [function(section, *args) for section in section_dict['section_list']]

    # Return the output
    return {
        'section_list': results,
        'section_height': section_dict['section_height'],
        'section_width': section_dict['section_width'],
        'height': section_dict['height'],
        'width': section_dict['width'],
    }

def show_image_sections(section_dict):
    # Calculate the number of rows and columns for the plot
    n_rows = int(np.sqrt(len(section_dict['section_list'])))
    n_cols = int(np.ceil(len(section_dict['section_list']) / n_rows))

    # Create a figure with subplots
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    ax = ax.ravel()

    # Plot each section in its own subplot
    for i, section in enumerate(section_dict['section_list']):
        ax[i].imshow(section)
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()