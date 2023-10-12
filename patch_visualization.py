import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data import closest_divisible_by_patch_size
from BeIT.MaskGenerator import MaskingGenerator

def visualize_image_patches(image_path, patch_size):
    """
    Visualize an image divided by patches.
    
    Parameters:
        image_path (str): Path to the image file.
        patch_size (tuple): A tuple (patch_height, patch_width) specifying the size of each patch.
    """
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    width = closest_divisible_by_patch_size(int(np.ceil(2100 * 0.4)))
    height = closest_divisible_by_patch_size(int(np.ceil(2970 * 0.4)))
    img = cv2.resize(img, (width, height))

    generator = MaskingGenerator()
    mask = generator(height//patch_size[0], width//patch_size[1])


    # Get image size
    img_height, img_width, _ = img.shape
    
    # Create a plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img)
    axs[1].imshow(img)

     # Loop through the image and mask grid
    for i in range(0, img_height, patch_size[0]):
        for j in range(0, img_width, patch_size[1]):
            # Check whether the patch is masked or not and choose the face color accordingly
            is_masked = mask[(i//patch_size[0])-1, (j//patch_size[1])-1]
            face_color = 'red' if is_masked else 'none'

            rect = patches.Rectangle(
                (j, i), 
                patch_size[1], 
                patch_size[0], 
                linewidth=0.5, 
                edgecolor='black',
                facecolor='none', 
                alpha=0.5  # Added alpha for some transparency
            )
            axs[0].add_patch(rect)
            
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (j, i), 
                patch_size[1], 
                patch_size[0], 
                linewidth=0.5, 
                edgecolor='black', 
                facecolor=face_color,
                alpha=0.5  # Added alpha for some transparency
            )
            axs[1].add_patch(rect)
    
    plt.savefig("patch_img.png")

# Example usage
visualize_image_patches('lg-334511307816371722-aug-beethoven--page-3.png', (64, 64))
