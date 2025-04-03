import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import requests
from pathlib import Path

# Set environment variables for compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

# Get Python version to determine compatible ultralytics version
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

# Install ultralytics if needed - try to get the latest compatible version
try:
    from ultralytics import YOLO
    print(f"Ultralytics already installed: {YOLO.__version__ if hasattr(YOLO, '__version__') else 'version unknown'}")
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    
    # For Python 3.12+, we need a newer version of ultralytics
    ultralytics_version = ">=8.0.0"
    if python_version.major == 3 and python_version.minor >= 12:
        print("Using Python 3.12+, installing latest ultralytics")
    else:
        print("Using Python <3.12, installing ultralytics 8.0.20")
        ultralytics_version = "==8.0.20"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"ultralytics{ultralytics_version}", "--no-cache-dir"])
        from ultralytics import YOLO
        print(f"Installed ultralytics: {YOLO.__version__ if hasattr(YOLO, '__version__') else 'version unknown'}")
    except Exception as e:
        print(f"Failed to install ultralytics: {e}")
        print("Trying alternative installation method...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--no-cache-dir", "--force-reinstall"])
            from ultralytics import YOLO
            print(f"Installed ultralytics via alternative method: {YOLO.__version__ if hasattr(YOLO, '__version__') else 'version unknown'}")
        except Exception as e2:
            print(f"All installation attempts failed: {e2}")
            sys.exit(1)

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
        print(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with YOLOv8 model."""
        print("Initializing ObjectDetector...")
        
        # First, ensure the model exists
        if not os.path.exists(MODEL_PATH):
            model_success = download_model(MODEL_PATH)
            if not model_success:
                raise RuntimeError(f"Could not download model to {MODEL_PATH}")
        else:
            print(f"Using existing model at {MODEL_PATH}")
        
        # Try loading the model with appropriate handling for different PyTorch versions
        print(f"PyTorch version: {torch.__version__}")
        model_loaded = False
        
        for attempt in range(3):
            try:
                print(f"Loading model - attempt {attempt+1}")
                if attempt == 0:
                    # Simple approach - should work with latest versions
                    self.model = YOLO(MODEL_PATH)
                elif attempt == 1:
                    # Specify task
                    self.model = YOLO(MODEL_PATH, task='detect')
                else:
                    # Specific for older PyTorch with newer Python
                    self.model = YOLO(MODEL_PATH, verbose=True)
                
                model_loaded = True
                print(f"Model loaded successfully on attempt {attempt+1}")
                break
            except Exception as e:
                print(f"Loading attempt {attempt+1} failed: {e}")
                
        if not model_loaded:
            print("Failed to load model after all attempts")
            raise RuntimeError("Could not load YOLOv8 model - incompatible versions")
        
        self.class_names = COCO_CLASSES
        print(f"Object detector initialized with PyTorch {torch.__version__}")
        
    def detect(self, frame):
        """Run object detection on a frame."""
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
        """Draw bounding boxes on the frame."""
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
        """Get a consistent color for a class."""
        # Generate a consistent color based on the class name
        hash_value = hash(class_name) % 255
        return (hash_value, 255 - hash_value, 100) 