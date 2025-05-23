import cv2
import numpy as np
from typing import List, Tuple
import base64

def crop_image(image: np.ndarray, box: List[Tuple[float, float]]) -> np.ndarray:
    """Crop image using normalized coordinates.
    
    Args:
        image: numpy array of image
        box: list of (x,y) tuples in normalized coordinates [0,1]
    
    Returns:
        Cropped image as numpy array
    """
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    box_pixels = [(int(x * w), int(y * h)) for x, y in box]
    
    # Get bounding rectangle
    x_coords = [x for x, y in box_pixels]
    y_coords = [y for x, y in box_pixels]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    
    # Add padding
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]

def to_bytes(image: np.ndarray) -> bytes:
    """Convert image to bytes.
    
    Args:
        image: numpy array of image
    
    Returns:
        Image encoded as bytes
    """
    success, encoded = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded.tobytes()

def preprocess_for_tf(image: np.ndarray) -> np.ndarray:
    """Preprocess image for TensorFlow models.
    
    Args:
        image: numpy array of image in BGR format
    
    Returns:
        Preprocessed image as numpy array
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to expected size
    image = cv2.resize(image, (224, 224))
    
    # Convert to float32 and normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    return image 