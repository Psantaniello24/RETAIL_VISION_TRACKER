import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import requests
from pathlib import Path

# Install ultralytics if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.20"])
    from ultralytics import YOLO

from collections import Counter
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, COCO_CLASSES, PRODUCTS

# Download YOLOv8 model if needed
def download_model(model_path):
    """Download YOLOv8 model if it doesn't exist"""
    if os.path.exists(model_path):
        return True
        
    print(f"Model not found at {model_path}, downloading...")
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with YOLOv8 model."""
        # Add compatibility fix for PyTorch version issues
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # First, ensure the model exists
        if not os.path.exists(MODEL_PATH):
            model_success = download_model(MODEL_PATH)
            if not model_success:
                raise RuntimeError(f"Could not download model to {MODEL_PATH}")
        
        # Try a few approaches to load the model
        for i in range(3):  # Three attempts
            try:
                if i == 0:
                    # First attempt: simple loading
                    self.model = YOLO(MODEL_PATH)
                    break
                elif i == 1:
                    # Second attempt: with task specified
                    self.model = YOLO(MODEL_PATH, task='detect')
                    break
                else:
                    # Third attempt: with weights_only=False
                    self.model = YOLO(MODEL_PATH, 
                                    _callbacks={'on_params_update': 
                                                lambda: torch.load(MODEL_PATH, weights_only=False)})
                    break
            except Exception as e:
                print(f"Loading attempt {i+1} failed: {e}")
                if i == 2:  # Last attempt
                    print("All loading attempts failed.")
                    raise
        
        self.class_names = COCO_CLASSES
        print(f"Model loaded successfully!")
        
    def detect(self, frame):
        """Run object detection on a frame.
        
        Args:
            frame: Input image/frame
            
        Returns:
            processed_frame: Frame with bounding boxes
            detections: Dictionary with counts of detected objects
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Get all detected objects
        detected_classes = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Only count products we're tracking
                if class_name in PRODUCTS:
                    detected_classes.append(class_name)
                    
        # Count occurrences of each class
        detection_counts = dict(Counter(detected_classes))
        
        # Process frame and draw boxes
        processed_frame = self._draw_boxes(frame, results[0])
        
        return processed_frame, detection_counts
    
    def _draw_boxes(self, frame, result):
        """Draw bounding boxes on the frame.
        
        Args:
            frame: Input image/frame
            result: YOLOv8 result object
            
        Returns:
            annotated_frame: Frame with bounding boxes
        """
        annotated_frame = frame.copy()
        
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            conf = box.conf[0]
            
            # Only draw boxes for products we're tracking
            if class_name in PRODUCTS:
                display_name = PRODUCTS[class_name]["name"]
                color = self._get_color(class_name)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                text = f"{display_name}: {conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Draw text
                cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _get_color(self, class_name):
        """Get a consistent color for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            color: BGR color tuple
        """
        # Generate a consistent color based on the class name
        hash_value = hash(class_name) % 255
        return (hash_value, 255 - hash_value, 100) 