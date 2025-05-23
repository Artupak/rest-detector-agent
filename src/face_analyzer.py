from typing import Dict, Optional, Tuple
import numpy as np
from deepface import DeepFace

class FaceAnalyzer:
    """Class for analyzing faces using DeepFace."""
    
    def __init__(self):
        """Initialize face analysis models."""
        # Models will be downloaded automatically on first use
        pass
        
    def analyze(self, face_img: np.ndarray) -> Dict[str, any]:
        """Analyze face for age and gender.
        
        Args:
            face_img: Face image as numpy array in BGR format
            
        Returns:
            Dictionary containing:
                - age: estimated age
                - gender: detected gender
                - gender_confidence: confidence score for gender
        """
        try:
            # Analyze face
            result = DeepFace.analyze(
                face_img,
                actions=['age', 'gender'],
                enforce_detection=False,
                silent=True
            )[0]
            
            return {
                'age': float(result['age']),
                'gender': result['gender'].lower(),
                'gender_confidence': float(result['gender_probability'])
            }
            
        except Exception as e:
            print(f"Error during face analysis: {str(e)}")
            return {
                'age': None,
                'gender': None,
                'gender_confidence': 0.0
            } 