from typing import Dict, List, Any, Optional
import json
import numpy as np
from .cursor import Cursor
from .detectors import ObjectDetector
from .face_analyzer import FaceAnalyzer
from .car_classifier import CarClassifier
from .utils import crop_image, preprocess_for_tf
from .config import Config

class DetectionAgent:
    """Main agent class that orchestrates the detection process."""
    
    def __init__(self, config: Config):
        """Initialize all components.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.cursor = Cursor(
            video_path=config.VIDEO, 
            image_folder=config.IMAGE_FOLDER,
            use_webcam=config.USE_WEBCAM,
            camera_id=config.CAMERA_ID
        )
        self.detector = ObjectDetector()
        self.face_analyzer = FaceAnalyzer()
        self.car_classifier = CarClassifier()
        
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a single frame.
        
        Args:
            frame: Image as numpy array
            
        Returns:
            List of detection results with classifications
        """
        if frame is None:
            return []
            
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Process each detection
        results = []
        for det in detections:
            # Crop detected object
            crop = crop_image(frame, det['box'])
            
            # Add base detection info
            result = {
                'name': det['name'],
                'score': det['score'],
                'box': det['box']
            }
            
            # Get specific predictions based on object type
            if det['name'] == 'Person':
                # Analyze face
                face_results = self.face_analyzer.analyze(crop)
                result.update(face_results)
            elif det['name'] in {'Car', 'Truck', 'Bus', 'Motorcycle'}:
                # Classify vehicle
                car_results = self.car_classifier.predict(crop)
                result.update(car_results)
            
            results.append(result)
            
        return results
    
    def run(self, progress_callback=None, display_callback=None) -> List[Dict[str, Any]]:
        """Run detection on all frames.
        
        Args:
            progress_callback: Optional callback function(progress: float)
            display_callback: Optional callback function(frame, results) for visualization
            
        Returns:
            List of all detection results
        """
        all_results = []
        
        while True:
            # Get next frame
            frame = self.cursor.next()
            if frame is None:
                break
                
            # Process frame
            frame_results = self.process_frame(frame)
            all_results.extend(frame_results)
            
            # Report progress
            if progress_callback:
                progress = self.cursor.get_progress()
                progress_callback(progress)
                
            # Display results if callback provided
            if display_callback:
                display_callback(frame, frame_results)
        
        return all_results 