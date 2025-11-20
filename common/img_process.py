import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def resize_obs(image, target_size=(96, 96)):
    """
    Crop the observation image and resize it to target_size.
    For CarRacing-v2: removes bottom info bar and resizes.
    
    Args:
        image: Raw observation image from environment (shape: 600x400x3 for CarRacing)
        target_size: Desired output dimensions as (width, height)
        
    Returns:
        numpy.ndarray: Processed image of shape (96, 96, 3) with uint8 dtype
    """
    # Convert to PIL Image
    img = Image.fromarray(image)
    
    # High-quality downsampling to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(img, dtype=np.uint8)


def preprocess_carracing_image(image, target_size=(96, 96), to_grayscale=True):
    """
    Full preprocessing pipeline for CarRacing images.
    
    Args:
        image: Raw observation from CarRacing environment (uint8 array)
        target_size: Target resolution (default: 96x96)
        to_grayscale: Whether to convert to grayscale (default: True)
        
    Returns:
        numpy.ndarray: Preprocessed image
            - If to_grayscale=True: shape (96, 96), dtype uint8, grayscale
            - If to_grayscale=False: shape (96, 96, 3), dtype uint8, RGB
    """
    # Remove bottom info bar (last 12 pixels from CarRacing-v2)
    image = image[:-12, :, :]

    processed = resize_obs(image, target_size)
    
    # Step 2: Convert to grayscale if requested
    if to_grayscale:
        img_pil = Image.fromarray(processed)
        img_pil = img_pil.convert('L')
        processed = np.array(img_pil, dtype=np.uint8)

    processed = transforms.ToTensor()(processed)  # Convert to tensor and normalize to [0, 1]

    return processed


def preprocess_breakout_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess Breakout observation image.
    - Downsampling to 96x96 from original resolution
    - Crop out top and bottom (game information)
    - Convert to grayscale
    
    Args:
        image: Raw observation image from Breakout environment
        
    Returns:
        torch.Tensor: Preprocessed image of shape (1, 96, 96) in grayscale
    """
    # Crop top and bottom to remove unrelated information
    # Typical Atari image is 210x160, crop top 20 pixels and bottom 20 pixels
    cropped = image[20:200, :, :]
    
    resized = resize_obs(cropped)
    img_pil = Image.fromarray(resized)
    
    # Convert to grayscale (1 channel)
    img_gray = img_pil.convert('L')
    
    # Convert to tensor and normalize to [0, 1]
    img_tensor = transforms.ToTensor()(img_gray)  # Shape: (1, 96, 96)
    
    return img_tensor