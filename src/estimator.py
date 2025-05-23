import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import Optional

class AgeEstimator:
    """Class for estimating age from face images using TensorFlow Hub model."""
    
    def __init__(self, model_path: str):
        """Initialize the age estimation model.
        
        Args:
            model_path: Path to TF Hub model
        """
        self.model = tf.keras.Sequential([
            hub.KerasLayer(model_path, input_shape=(224, 224, 3))
        ])
    
    def predict(self, face_img: np.ndarray) -> Optional[float]:
        """Predict age from face image.
        
        Args:
            face_img: Preprocessed face image as numpy array (224,224,3) in [0,1] range
            
        Returns:
            Estimated age as float, or None if prediction fails
        """
        try:
            # Add batch dimension
            face_img_batch = face_img[np.newaxis, ...]
            
            # Get prediction
            prediction = self.model.predict(face_img_batch, verbose=0)
            
            # Extract age value
            age = float(prediction[0][0])
            
            return age
            
        except Exception as e:
            print(f"Error during age prediction: {str(e)}")
            return None 