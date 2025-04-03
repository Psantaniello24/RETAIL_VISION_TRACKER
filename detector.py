import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys

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

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with YOLOv8 model."""
        # Add compatibility fix for PyTorch version issues
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        try:
            # Handle PyTorch 2.6+ compatibility
            # Directly use weights_only=False which is the safest approach
            self.model = YOLO(MODEL_PATH, task='detect')
            
        except Exception as e:
            print(f"First attempt failed: {e}")
            print("Trying alternate approach...")
            
            try:
                # Patch torch.load to always use weights_only=False
                original_torch_load = torch.load
                
                def patched_load(f, *args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(f, *args, **kwargs)
                
                torch.load = patched_load
                
                # Try loading with the patch
                self.model = YOLO(MODEL_PATH)
                
                # Restore original function
                torch.load = original_torch_load
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Loading failed. Please check model path and dependencies.")
                raise
        
        self.class_names = COCO_CLASSES
        
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