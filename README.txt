This repo is a compendium of the techniques implemented in image processing.

Section 1 involves the foundations of image arithmetic and processing concepts.
Section 2 involves classical edge detection and image segementation techniques.
Section 3 involves the final project, creating custom features for an SVM, and exploiting pyramidal decomposition to better deal with variance in scale of targets in the images.

There is also the code for a research presentation. This code was an exploration into stacking segmentation techniques in order to track contiguous lines in images. Though interesting, it wasn't particularly effective. Perhaps it can be improved with more tuning and techniques.

ImageLab is an intermediate attempt to compile the functions into a library. However, I found that using cv2 was often far more efficient, since Python isn't a suitable language for handling low level computation efficiently. My later work directly utilized libraries and their respective functions due their greater efficiency.
