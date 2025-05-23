import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import Optional, Tuple

class GenderDetector:
    """Class for detecting gender from face images using TensorFlow Hub model."""
    
    def __init__(self, model_path: str = "https://tfhub.dev/google/tfjs-model/gender-detection/1"):
        """Initialize the gender detection model.
        
        Args:
            model_path: Path to TF Hub model
        """
        self.model = tf.keras.Sequential([
            hub.KerasLayer(model_path, input_shape=(224, 224, 3))
        ])
        
    def predict(self, face_img: np.ndarray) -> Tuple[str, float]:
        """Predict gender from face image.
        
        Args:
            face_img: Preprocessed face image as numpy array (224,224,3) in [0,1] range
            
        Returns:
            Tuple of (gender, confidence) where gender is 'male' or 'female'
        """
        try:
            # Add batch dimension
            face_img_batch = face_img[np.newaxis, ...]
            
            # Get prediction
            prediction = self.model.predict(face_img_batch, verbose=0)
            
            # Convert to probability
            prob = float(prediction[0][0])
            
            # Classify based on probability
            gender = 'male' if prob > 0.5 else 'female'
            confidence = prob if gender == 'male' else 1 - prob
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error during gender prediction: {str(e)}")
            return None, 0.0 