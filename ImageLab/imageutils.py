import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image

__all__ = ["ImageUtil", "ImagePlotter", "MultiPlotter"]

class ImageUtil:
    # Deals with PIL objects
    
    def __init__(self, img):
        self.img = np.uint8(img)
    
    def save_image_to_folder(self, folder_name, filename):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Convert image to the appropriate color space if necessary
        if len(self.img.shape) == 2:
            mode = 'L'
        elif self.img.shape[2] == 3:
            mode = 'RGB'
        elif self.img.shape[2] == 1:
            self.img = np.squeeze(self.img)
            mode = 'L'
        elif self.img.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError('Unsupported image format')
        
        
        if mode == 'RGB':
            pil_img = Image.fromarray(self.img)
        elif mode == 'L':
            pil_img = Image.fromarray(self.img, mode=mode)
        elif mode == 'RGBA':
            pil_img = Image.fromarray(self.img, mode=mode)
            pil_img = pil_img.convert('RGB').convert('RGBA')

        # Save the image to the new folder under the same name
        new_path = os.path.join(folder_name, filename)
        pil_img.save(new_path)

        return new_path
    
    def load_image(self, folder, file):
        
        filepath = os.path.join(folder, file)
        
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

    def load_image_from_file(self, filepath):
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
    
    def np_to_PIL_convert(self, img):
    # Convert numpy type to PIL image
        output = Image.fromarray(np.uint8(img))
        self.img = output
        return output

    def plot_images_in_folder(self, folder_path, cmap=None):
        # Get list of image filenames in the folder, sorted alphabetically
        image_filenames = sorted([filename for filename in os.listdir(folder_path)
                              if filename.endswith('.jpg') or filename.endswith('.png')])
        
        
        # Loop through all files in the folder
        for filename in image_filenames:
            # Open the image file using PIL
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                name = os.path.splitext(filename)[0]
                if cmap == None:
                    ImagePlotter(np.asarray(img)).plot_image_with_histogram(
                        f'{name}')
                else:
                    # Apply the input function to the image
                    ImagePlotter(np.asarray(img)).plot_image_with_histogram(f'{name}', cmap=cmap)

    def get_image_type(self, rgb = True):
        # Check the number of dimensions of the image array
        if self.img.ndim == 2:
            return 'Grayscale'
        elif self.img.ndim == 3:
            # Check the shape of the third dimension
            if self.img.shape[2] == 3 and rgb == True:
                return 'RGB'
            elif self.img.shape[2] == 1:
                return 'Grayscale'
            elif self.img.shape[2] == 4:
                return 'RGBA'
            elif self.img.shape[2] == 2:
                return 'grayscale_alpha'
            elif self.img.shape[2] == 3 and rgb == False:
                return 'HSV'
        else:
            return 'unknown'

class ImagePlotter(ImageUtil):
    def __init__(self, img):
        super().__init__(img)
        
    def plot_HSV(self):
        # Separate the layers 0, 1, and 2
        H = self.img[:, :, 0]
        S = self.img[:, :, 1]
        V = self.img[:, :, 2]

        # Plot the channels and their histograms
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

        # Set the spacing between the subplots
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

        # Convert the HSV image to RGB for display
        rgb_img = cv2.cvtColor(
            cv2.convertScaleAbs(self.img), cv2.COLOR_HSV2RGB)
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('Original')

        axes[1, 0].hist(self.img.flatten(), bins=256,
                        range=(0, 256), color='gray')
        axes[1, 0].set_title('Original histogram')

        axes[0, 1].imshow(H, cmap='hsv')
        axes[0, 1].set_title('Hue')

        axes[1, 1].hist(H.flatten(), bins=256,
                        range=(0, 256), color='gray')
        axes[1, 1].set_title('Hue histogram')

        axes[0, 2].imshow(S, cmap='Greys')
        axes[0, 2].set_title('Saturation')

        axes[1, 2].hist(S.flatten(), bins=256,
                        range=(0, 256), color='gray')
        axes[1, 2].set_title('Saturation histogram')

        axes[0, 3].imshow(V, cmap='Greys')
        axes[0, 3].set_title('Value')

        axes[1, 3].hist(V.flatten(), bins=256,
                        range=(0, 256), color='gray')
        axes[1, 3].set_title('Value histogram')
        
        plt.tight_layout()
        plt.show()

    def plot_RGB(self):
        # Separate the red, green, and blue channels
        r_channel = self.img[:, :, 2]
        g_channel = self.img[:, :, 1]
        b_channel = self.img[:, :, 0]

        # Plot the red, green, and blue channels and their histograms
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

        # Set the spacing between the subplots
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

        axes[0, 0].imshow(self.img)
        axes[0, 0].set_title('Original')

        axes[1, 0].hist(self.img.flatten(), bins=256,
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
        
    def plot_image_with_histogram(self, title, plot_text = None):
        
        # Calculate the image histogram
        histogram, bin_edges = np.histogram(self.img.flatten(), bins=256, range=(0, 256))

        # Create a new figure with two subplots: one for the image and one for the histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        img_type = ImageUtil(self.img).get_image_type()
        
        if img_type == 'Grayscale':
            ax1.imshow(self.img, cmap='gray')
        if img_type == 'RGB':
            ax1.imshow(self.img)
        if img_type == 'HSV':
            ax1.imshow(self.img, cmap='hsv')
            
        ax1.set_title(title)

        # Plot the histogram in the second subplot
        ax2.bar(bin_edges[:-1], histogram, width=1)
        ax2.set_title('{} Histogram'.format(title))
        
        # Calculate the midpoint value, mean, and variance of the histogram
        mode = np.argmax(histogram)
        mean = np.mean(self.img)
        variance = np.var(self.img)

        # Add the midpoint value, mean, and variance to the plot
        ax2.axvline(x=mode, color='r', linestyle='--',
                        label=f"Mode: {mode:.2f}")
        ax2.axvline(x=mean, color='g', linestyle='--',
                        label=f"Mean: {mean:.2f}")
        ax2.axvline(x=mean-np.sqrt(variance), color='b',
                        linestyle='--', label=f"Variance: {variance:.2f}")
        ax2.axvline(x=mean+np.sqrt(variance), color='b', linestyle='--')
        ax2.legend()
        
        # Add plot text below the plot
        if plot_text is not None:
            plt.gcf().text(0.5, 0.01, plot_text, ha='center', fontsize=12)

        # Display the image and histogram
        plt.show()
        
    def plot_image(self, title):

        # Create a new figure with two subplots: one for the image and one for the histogram
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        img_type = ImageUtil(self.img).get_image_type()

        if img_type == 'Grayscale':
            ax1.imshow(self.img, cmap='gray')
        if img_type == 'RGB':
            ax1.imshow(self.img)
        if img_type == 'HSV':
            ax1.imshow(self.img, cmap='hsv')

        ax1.set_title(title)

    
class MultiPlotter(ImageUtil):
    #Takes Pillow Objects
    def __init__(self, image_list, name = 'default'):
        self.images = image_list
        
    def plot_images_with_histograms(self, title = 'default', cmap = None):
        for i, img in enumerate(self.images):
            ImagePlotter(img).plot_image_with_histogram(title=f'{title} {i}', cmap=cmap)