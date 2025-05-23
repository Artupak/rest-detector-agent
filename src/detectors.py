from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    """Class for detecting objects using YOLOv8."""
    
    def __init__(self):
        """Initialize the YOLO model."""
        self.model = YOLO('yolov8n.pt')  # Use nano model for speed
        self.target_objects = {
            'person': 'Person',
            'car': 'Car',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle',
            'bicycle': 'Bicycle'
        }
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in an image.
        
        Args:
            frame: Image as numpy array in BGR format
            
        Returns:
            List of dictionaries containing detection results with fields:
                - name: object class name
                - score: confidence score
                - box: list of (x,y) coordinates for bounding box
        """
        # Run inference
        results = self.model.predict(frame, verbose=False)[0]
        
        # Process results
        detections = []
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            class_name = results.names[int(cls)]
            
            # Check if this is a target object
            if class_name in self.target_objects:
                # Convert to normalized coordinates
                h, w = frame.shape[:2]
                box = [
                    (float(x1)/w, float(y1)/h),
                    (float(x2)/w, float(y1)/h),
                    (float(x2)/w, float(y2)/h),
                    (float(x1)/w, float(y2)/h)
                ]
                
                detections.append({
                    'name': self.target_objects[class_name],
                    'score': float(conf),
                    'box': box
                })
        
        return detections 