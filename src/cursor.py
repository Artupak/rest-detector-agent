import os
import cv2
from typing import Optional, Union
import numpy as np
from PIL import Image

class Cursor:
    def __init__(self, video_path: Optional[str] = None, 
                 image_folder: Optional[str] = None,
                 use_webcam: bool = False,
                 camera_id: int = 0):
        """Initialize cursor for video, image folder, or webcam processing.
        
        Args:
            video_path: Path to video file
            image_folder: Path to folder containing images
            use_webcam: Whether to use webcam
            camera_id: Camera device ID (usually 0 for built-in webcam)
        """
        sources = sum([bool(video_path), bool(image_folder), use_webcam])
        if sources > 1:
            raise ValueError("Can only specify one source: video_path, image_folder, or webcam")
        if sources == 0:
            raise ValueError("Must specify one source: video_path, image_folder, or webcam")
            
        self.frame_count = 0
        
        if use_webcam:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera with ID {camera_id}")
            self.mode = 'webcam'
            self.total_frames = float('inf')  # Webcam has infinite frames
        elif video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.mode = 'video'
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.folder = image_folder
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.images = [f for f in sorted(os.listdir(image_folder)) 
                         if f.lower().endswith(valid_extensions)]
            self.index = 0
            self.mode = 'images'
            self.total_frames = len(self.images)
    
    def next(self) -> Optional[np.ndarray]:
        """Get next frame from video, image from folder, or webcam.
        
        Returns:
            numpy array of image in BGR format, or None if no more frames
        """
        self.frame_count += 1
        
        if self.mode in ['video', 'webcam']:
            ret, frame = self.cap.read()
            if self.mode == 'webcam' and not ret:
                # For webcam, try to reconnect if frame grab fails
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                ret, frame = self.cap.read()
            return frame if ret else None
        else:
            if self.index < len(self.images):
                img_path = os.path.join(self.folder, self.images[self.index])
                self.index += 1
                return cv2.imread(img_path)
            return None
    
    def get_progress(self) -> float:
        """Get progress as percentage."""
        if self.mode == 'webcam':
            return 0.0  # No progress for webcam
        return (self.frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode in ['video', 'webcam']:
            self.cap.release() 