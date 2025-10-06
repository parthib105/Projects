import matplotlib.pyplot as plt
import os

def save_comparison_plot(images, titles, main_title, save_path):
    """
    Creates and saves a comparison plot for segmentation results.
    Args:
        images (list): A list of images (numpy arrays) to display.
        titles (list): A list of titles corresponding to the images.
        main_title (str): The main title for the entire figure.
        save_path (str): The full path to save the output image file.
    """
    if len(images) != len(titles):
        raise ValueError("The number of images must match the number of titles.")

    n = len(images)
    plt.figure(figsize=(4 * n, 4))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close the figure to free up memory