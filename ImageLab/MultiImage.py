from PIL import Image
from .imageutils import ImageUtil, ImagePlotter
import numpy as np
from numba import jit

__all__ = ['MultiImage', 'Gradient_Magnitude', 'Gradient_Direction']


class MultiImage:
    def __init__(self, image1_path, image2_path, folder_name, output_name, hist = False):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.folder_name = folder_name
        self.img_name = output_name
        self.hist = hist
    
    def process(self, operator):
        # Open the input image
        himg = Image.open(self.image1_path)
        vimg = Image.open(self.image2_path)
        himg = np.array(himg)
        vimg = np.array(vimg)
        
        
        if himg.ndim == 2:
            himg = np.expand_dims(himg, axis=2)
            
        if vimg.ndim == 2:
            vimg = np.expand_dims(vimg, axis=2)
        
        # Apply the operator to the input image
        output = operator.apply(himg, vimg)
        
        output = np.clip(output, 0, 255).astype(np.uint8)

        path = ImageUtil(output).save_image_to_folder(
            f'Image/{self.folder_name}/', f"{self.img_name}.png")
        
        if self.hist == True:
            ImagePlotter(output).plot_image_with_histogram(f'{self.img_name}')
            
        else:
            ImagePlotter(output).plot_image(f'{self.img_name}')
        
        return output, path


class Gradient_Magnitude:
    def __init__(self):
        pass
    
    def apply(self, himg, vimg):
        mag = np.zeros_like(himg).astype(np.float64)
        return self.magnitude(mag,himg,vimg).astype(np.uint8)
    
    @staticmethod
    @jit(nopython=True)
    def magnitude(mag, himg, vimg):
        for i in range(himg.shape[0]):
            for j in range(himg.shape[1]):
                mag[i][j] = (himg[i][j]**2 + vimg[i][j]**2)**0.5
        return (mag / np.amax(mag)) * 255.0

class Gradient_Direction:
    def __init__(self):
        pass
    def apply(self, himg, vimg):
        angle = np.arctan2(vimg.astype(np.float32),
                           himg.astype(np.float32))  # compute angle

        # Convert the radians to degrees and scale the range from [0, pi] to [0, 255]
        return ((angle + np.pi) * 255.0 / (2 * np.pi))
    
class Difference:
    def __init__(self, ratio = [1,1]):
        self.ratio = ratio
    def apply(self, himg, vimg):
        return himg * self.ratio[0] - vimg * self.ratio[1]
    
class Sum:
    def __init__(self, ratio = [1,1]):
        self.ratio = ratio
    def apply(self, himg, vimg):
        return himg * self.ratio[0] + vimg * self.ratio[1]
        
def hough_transform(img_bin, theta_res=1, rho_res=1):
    h, w = img_bin.shape
    diag_len = int(np.ceil(np.sqrt(h*h + w*w)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 / rho_res + 1)
    thetas = np.arange(0, 180, theta_res)

    cos_t = np.cos(np.deg2rad(thetas))
    sin_t = np.sin(np.deg2rad(thetas))
    num_thetas = len(thetas)

    accumulator = np.zeros(
        (int(2 * diag_len / rho_res), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img_bin)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int((x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos
