from google.cloud import aiplatform
from typing import Dict, List, Any
import numpy as np

class VehicleModelClassifier:
    """Class for classifying vehicle models using Google Cloud AutoML Vision."""
    
    def __init__(self, endpoint_id: str, project: str, location: str):
        """Initialize the AutoML Vision endpoint.
        
        Args:
            endpoint_id: ID of the deployed model endpoint
            project: Google Cloud project ID
            location: Google Cloud region
        """
        aiplatform.init(project=project, location=location)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
    
    def predict(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Predict vehicle model from image.
        
        Args:
            image_bytes: Image encoded as bytes
            
        Returns:
            List of dictionaries containing:
                - displayNames: list of predicted model names
                - scores: corresponding confidence scores
        """
        instance = {"content": image_bytes}
        prediction = self.endpoint.predict(instances=[instance])
        
        if not prediction.predictions:
            return []
            
        # Format response
        results = []
        for pred in prediction.predictions:
            results.append({
                'displayNames': pred.get('displayNames', []),
                'scores': pred.get('scores', [])
            })
            
        return results 