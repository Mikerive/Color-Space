import numpy as np

class NoiseEstimator:
    def __init__(self, img):
        self.img = np.asarray(img)
        if np.ndim(self.img) == 2:
            self.img = np.expand_dims(self.img, axis=2)
            
    # Estimates pepper noise on a single layer
    def pepper_noise_estimator(self, layer = [0,1,2]):
        if self.img.shape[2]==1:
            layer = [0]
        
        # Density to be returned for every layer
        density = np.zeros(len(layer))
        
        for i in layer:
            count = np.count_nonzero(np.logical_or(
                self.img[:, :, i] < 4, self.img[:, :, i] > 251))
            # density
            density[i] = count / self.img[:,:,i].size
        
        for i in range(len(layer)):
            print(f"Layer {layer[i]}: density = {density[i]}")
        
        return density

    # Estimates gaussian noise on all layers
    def gaussian_mean_var_estimator(self, layer = [0,1,2]):
        
        if self.img.shape[2] == 1:
            layer = [0]
        
        mean = np.array([])
        var = np.array([])
        
        for i in layer:
            image = self.img[:,:,i]
            # Calculate the histogram
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            
            mean = np.append(mean, np.mean(hist))
            var = np.append(var, np.var(hist))
            
            # Print the mean and variance values for each layer
        for i in range(len(layer)):
            print(f"Layer {layer[i]}: Mean = {mean[i]}, Variance = {var[i]}")
        
        return mean, var
