import os
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, video_path=None, image_folder=None, use_webcam=False, camera_id=0):
        # Input source
        self.VIDEO = video_path
        self.IMAGE_FOLDER = image_folder
        self.USE_WEBCAM = use_webcam
        self.CAMERA_ID = camera_id
        
        # Display settings
        self.DISPLAY_SCALE = 1.0  # Scale factor for display window
        self.SHOW_CONFIDENCE = True  # Whether to show confidence scores
        self.FONT_SCALE = 0.5  # Text size for annotations 