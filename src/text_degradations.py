# text_degradations.py

import cv2
import numpy as np
import random

def create_text_mask(clean_image, threshold=220):
    """
    Creates a binary mask of the text from a clean image.
    Assumes dark text on a light background.

    Args:
        clean_image (np.array): The original clean Manchu image.
        threshold (int): The grayscale value to separate text from background.

    Returns:
        np.array: A 2D binary mask where text is 255 (white) and background is 0 (black).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    
    # Invert the thresholding: we want text to be white (the "active" area)
    _, text_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return text_mask

# 移除 apply_morphology
# def apply_morphology(image, text_mask):
#     """
#     Simulates ink bleeding (dilate) or fading (erode) on the text.
#     This effect is applied to the entire image but then masked to only affect text areas.
#     
#     Args:
#         image (np.array): The fused image to apply degradation on.
#         text_mask (np.array): The mask identifying text regions.
#         
#     Returns:
#         np.array: The image with morphological effects applied to the text.
#     """
#     # Create a random kernel
#     kernel_size = random.choice([2, 3])
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     
#     # Choose to either erode (fade) or dilate (bleed)
#     if random.random() < 0.5:
#         morphed_image = cv2.erode(image, kernel, iterations=1)
#     else:
#         morphed_image = cv2.dilate(image, kernel, iterations=1)
#         
#     # Create a copy of the original image to preserve the background
#     output_image = image.copy()
#     
#     # Use the text mask to copy the morphed text onto the output image
#     output_image[text_mask > 0] = morphed_image[text_mask > 0]
#     
#     return output_image

# 移除 apply_text_blur
# def apply_text_blur(image, text_mask):
#     """
#     Applies Gaussian blur specifically to the text areas to simulate loss of sharpness.
#     
#     Args:
#         image (np.array): The fused image.
#         text_mask (np.array): The mask identifying text regions.
#         
#     Returns:
#         np.array: The image with blurred text.
#     """
#     # Get a random kernel size (must be an odd number)
#     kernel_size = random.choice([3, 5])
#     
#     # Blur the entire image
#     blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
#     
#     # Create a copy to preserve the background
#     output_image = image.copy()
#     
#     # Use the text mask to copy the blurred text onto the output image
#     output_image[text_mask > 0] = blurred_image[text_mask > 0]
#     
#     return output_image

def apply_pepper_noise_to_text(image, text_mask, background, amount=0.05):
    """
    Simulates ink flaking off by replacing random text pixels with the background.
    This is an "erasing" effect.
    
    Args:
        image (np.array): The fused image to apply degradation on.
        text_mask (np.array): The mask identifying text regions.
        background (np.array): The background texture patch to reveal.
        amount (float): The percentage of text pixels to erase.
        
    Returns:
        np.array: The image with parts of the text erased.
    """
    output_image = image.copy()
    
    # Find the coordinates of all text pixels
    text_coords = np.argwhere(text_mask > 0)
    
    # Determine the number of pixels to erase
    num_pixels_to_erase = int(len(text_coords) * amount)
    
    # Choose random indices to erase
    erase_indices = np.random.choice(len(text_coords), num_pixels_to_erase, replace=False)
    
    # Get the coordinates to erase
    erase_coords = text_coords[erase_indices]
    
    # Replace the pixels in the output image with pixels from the background
    # This uses advanced NumPy indexing for speed
    # erase_coords is a list of [row, col], so we need to transpose it for indexing
    rows, cols = erase_coords[:, 0], erase_coords[:, 1]
    output_image[rows, cols] = background[rows, cols]
    
    return output_image