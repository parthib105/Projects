import numpy as np
from skimage import io, color, img_as_float

def load_image_gray(path: str, normalize=True):
    """Loads an image from a path and converts it to grayscale float."""
    try:
        img = io.imread(path)
        if img.ndim == 3:
            if img.shape[2] == 4: # Handle RGBA
                img = color.rgba2rgb(img)
            img = color.rgb2gray(img)
        if normalize:
            img = img_as_float(img)
        return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return None
    
def kapur_entropy_threshold(image_hist, n_pixels):
    """
    Calculates the optimal threshold for an image using Kapur's maximum entropy method.
    Args:
        image_hist (np.array): Normalized histogram of the image.
        n_pixels (int): Total number of pixels in the image.
    Returns:
        float: The optimal threshold, normalized to the range [0, 1].
    """
    L = len(image_hist)
    max_entropy = -np.inf
    best_threshold = 0

    # Calculate total entropy of the image histogram
    Hn = 0
    for p_i in image_hist:
        if p_i > 0:
            Hn -= p_i * np.log(p_i)

    for t in range(1, L):
        # Probability of object (foreground)
        P_obj = np.sum(image_hist[:t])
        # Probability of background
        P_bkg = 1.0 - P_obj

        if P_obj <= 1e-6 or P_bkg <= 1e-6:
            continue

        # Entropy of object
        H_obj = 0
        for i in range(t):
            if image_hist[i] > 0:
                H_obj -= (image_hist[i] / P_obj) * np.log(image_hist[i] / P_obj)

        # Entropy of background
        H_bkg = 0
        for i in range(t, L):
            if image_hist[i] > 0:
                H_bkg -= (image_hist[i] / P_bkg) * np.log(image_hist[i] / P_bkg)

        current_entropy = H_obj + H_bkg
        if current_entropy > max_entropy:
            max_entropy = current_entropy
            best_threshold = t

    return best_threshold / (L - 1)  # Normalize threshold to [0, 1]