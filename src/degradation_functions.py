# -*- coding: utf-8 -*-

# ============================================================================
# DEGRADATION FUNCTIONS LIBRARY
# This file contains all standalone degradation effect functions.
# Each function should take a numpy array (image) as input
# and return a numpy array (processed image).
# ============================================================================

import cv2
import numpy as np
import random

def apply_blur(image):
    """Applies a random Gaussian blur to the image."""
    kernel_size = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 移除 apply_ink_erosion 函数
# def apply_ink_erosion(image):
#     """Simulates ink erosion or bleeding using morphological operations."""
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (random.randint(2,3), random.randint(2,3)))
#     op = random.choice([cv2.erode, cv2.dilate])
#     return op(image, kernel, iterations=1)

def apply_jpeg_artifacts(image):
    """Adds JPEG compression artifacts."""
    quality = random.randint(40, 85)
    result, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if result:
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        return decoded_image
    return image # Return original if encoding fails

def apply_random_brightness_contrast(image):
    """Adjusts brightness and contrast randomly."""
    alpha = 1.0 + random.uniform(-0.3, 0.3)  # Contrast control
    beta = random.randint(-20, 20)          # Brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# --- Add more degradation functions below ---
# def apply_pepper_noise(...)
# def ...