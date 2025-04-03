import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import requests
from pathlib import Path
import time

# Install ultralytics if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.45"])
    from ultralytics import YOLO

from collections import Counter
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, COCO_CLASSES, PRODUCTS,
    CPU_OPTIMIZED, CPU_IMGSZ, GPU_IMGSZ
)

# Download YOLOv8 model if needed
def download_model(model_path):
    """Download YOLOv8 model if it doesn't exist"""
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return True
        
    print(f"Model not found at {model_path}, downloading...")
    
    # Multiple fallback URLs to try in order
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt",
        "https://huggingface.co/ultralytics/yolov8n/resolve/main/yolov8n.pt"
    ]
    
    # Create directories if needed
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory: {model_dir}")
    
    # Try each URL until one works
    for i, url in enumerate(urls):
        print(f"Download attempt {i+1}/{len(urls)} from: {url}")
        try:
            # Set a reasonable timeout
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size/1024/1024:.1f} MB")
            
            # Download with progress tracking
            downloaded = 0
            start_time = time.time()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 1MB
                        if downloaded % (1024*1024) < 8192:
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = downloaded / elapsed / 1024 / 1024
                                percent = 100 * downloaded / total_size if total_size > 0 else 0
                                print(f"Downloaded: {downloaded/1024/1024:.1f}MB ({percent:.1f}%) - {speed:.1f} MB/s")
            
            # Verify file was downloaded
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                print(f"Successfully downloaded model to {model_path}")
                return True
            else:
                print("Download completed but file is empty, trying next URL")
        except Exception as e:
            print(f"Error during download from {url}: {e}")
    
    # If we get here, all download attempts failed
    print("All download attempts failed")
    
    # Last resort: try using ultralytics to download the model directly
    try:
        print("Attempting to use ultralytics to download the model directly...")
        # This will download the model to the ultralytics cache directory
        model = YOLO("yolov8n.pt")
        
        # Try to copy the model to our target location
        import shutil
        model_files = sorted(Path(os.path.expanduser("~/.cache")).rglob("yolov8n*.pt"))
        if model_files:
            shutil.copy(str(model_files[0]), model_path)
            print(f"Copied model from cache: {model_files[0]} -> {model_path}")
            return True
    except Exception as e:
        print(f"Failed to use ultralytics to download the model: {e}")
    
    return False

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with YOLOv8 model."""
        # Add compatibility fix for PyTorch version issues
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Set image size based on device and optimization settings
        if self.device == 'cpu' and CPU_OPTIMIZED:
            self.default_imgsz = CPU_IMGSZ  # Smaller size for CPU
            print(f"CPU optimization active - using smaller image size: {self.default_imgsz}")
        else:
            self.default_imgsz = GPU_IMGSZ  # Default size
            print(f"Using standard image size: {self.default_imgsz}")
        
        # Add threading optimization for CPU
        if self.device == 'cpu' and CPU_OPTIMIZED:
            torch.set_num_threads(max(1, torch.get_num_threads() - 1))
            print(f"Set PyTorch threads to {torch.get_num_threads()} for CPU optimization")
        
        # First, ensure the model exists
        if not os.path.exists(MODEL_PATH):
            model_success = download_model(MODEL_PATH)
            if not model_success:
                # Try one last approach - use a model built into ultralytics
                try:
                    print("Attempting to use a pre-cached model from ultralytics...")
                    self.model = YOLO("yolov8n.pt")  # This should trigger a download if needed
                    print("Successfully loaded model from ultralytics cache")
                except Exception as e:
                    print(f"All model loading attempts failed: {e}")
                    raise RuntimeError(f"Could not download model to {MODEL_PATH}")
                return
        
        # Modern PyTorch-compatible loading approach
        try:
            print(f"Loading YOLO model with {self.device}...")
            # Specify device and image size for optimal performance
            self.model = YOLO(MODEL_PATH)
            
            # Perform a warmup inference to initialize the model
            print("Running model warmup...")
            dummy_img = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
            self.model(dummy_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
            print("Model warmup completed")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting fallback loading methods...")
            
            try:
                # Try with task specification
                print("Trying with task specification...")
                self.model = YOLO(MODEL_PATH, task='detect')
            except Exception as inner_e:
                print(f"Task-specific loading failed: {inner_e}")
                
                # Last resort - try to use a pre-cached model
                try:
                    print("Attempting to use a pre-cached model from ultralytics...")
                    self.model = YOLO("yolov8n.pt")  # This should trigger a download if needed
                    print("Successfully loaded model from ultralytics cache")
                except Exception as final_e:
                    print(f"Final attempt failed: {final_e}")
                    raise RuntimeError("Could not load YOLOv8 model after multiple attempts")
        
        self.class_names = COCO_CLASSES
        print(f"Model loaded successfully!")
        
    def detect(self, frame, imgsz=None):
        """Run object detection on a frame."""
        start_time = time.time()
        
        # Use specified image size or default
        imgsz = imgsz or self.default_imgsz
        
        # Run YOLOv8 inference with specified size
        try:
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=imgsz, verbose=False)
        except:
            # Fallback without imgsz parameter if it fails
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Get all detected objects
        detected_classes = []
        for result in results:
            # Check if we're using newer or older ultralytics API
            try:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    
                    # Only count products we're tracking
                    if class_name in PRODUCTS:
                        detected_classes.append(class_name)
            except AttributeError:
                # Newer API might have different structure
                try:
                    # Try newer ultralytics API format
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = self.class_names[class_id]
                        
                        # Only count products we're tracking
                        if class_name in PRODUCTS:
                            detected_classes.append(class_name)
                except:
                    # Last resort: try to extract data from results in any way possible
                    if hasattr(result, 'names') and hasattr(result, 'boxes'):
                        for box in result.boxes:
                            if hasattr(box, 'cls'):
                                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                                if hasattr(result, 'names'):
                                    class_name = result.names[class_id]
                                else:
                                    class_name = self.class_names[class_id]
                                
                                if class_name in PRODUCTS:
                                    detected_classes.append(class_name)
                    
        # Count occurrences of each class
        detection_counts = dict(Counter(detected_classes))
        
        # Process frame and draw boxes
        processed_frame = self._draw_boxes(frame, results[0])
        
        # Log performance if in CPU mode
        if self.device == 'cpu' and CPU_OPTIMIZED:
            inference_time = time.time() - start_time
            if inference_time > 0.1:  # Only log slower operations
                print(f"Detection took {inference_time:.2f}s on CPU at size {imgsz}")
        
        return processed_frame, detection_counts
    
    def _draw_boxes(self, frame, result):
        """Draw bounding boxes on the frame."""
        annotated_frame = frame.copy()
        
        try:
            # Try the traditional approach first
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
        except (AttributeError, IndexError) as e:
            print(f"Error in original drawing method: {e}")
            
            try:
                # Fallback to using the plot method from ultralytics
                annotated_frame = result.plot()
            except Exception as e2:
                print(f"Error in fallback drawing method: {e2}")
                # Just return the original frame if both methods fail
                pass
        
        return annotated_frame
    
    def _get_color(self, class_name):
        """Get a consistent color for a class."""
        # Generate a consistent color based on the class name
        hash_value = hash(class_name) % 255
        return (hash_value, 255 - hash_value, 100) 