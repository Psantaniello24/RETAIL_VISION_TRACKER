# YOLOv8 Configuration
MODEL_PATH = "yolov8n.pt"  # Using the default YOLOv8 nano model
CONFIDENCE_THRESHOLD = 0.4  # Lowered slightly to detect more objects

# Webcam Configuration
CAMERA_ID = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Video File Configuration
VIDEO_PATH = "demo_video.mp4"  # Default demo video file path

# Performance Configuration
# Set to True to enable CPU optimizations
CPU_OPTIMIZED = True
# Default CPU image size (smaller = faster)
CPU_IMGSZ = 320
# Default GPU image size
GPU_IMGSZ = 640
# Default skip frames for CPU (higher = faster but less accuracy)
CPU_SKIP_FRAMES = 2
# Default skip frames for GPU
GPU_SKIP_FRAMES = 0

# Inventory Configuration
PRODUCTS = {
    "person": {"name": "Customer", "threshold": 2, "alert": True}, 
    "bottle": {"name": "Beverage", "threshold": 3, "alert": True},
    "cup": {"name": "Cup", "threshold": 3, "alert": True},
    "bowl": {"name": "Bowl", "threshold": 2, "alert": True},
    "chair": {"name": "Chair", "threshold": 1, "alert": True},
    "book": {"name": "Book", "threshold": 3, "alert": True},
    "cell phone": {"name": "Phone", "threshold": 2, "alert": True},
    "wine glass": {"name": "Glass", "threshold": 2, "alert": True},
    "tie": {"name": "Tie", "threshold": 1, "alert": True},
    "suitcase": {"name": "Suitcase", "threshold": 1, "alert": True},
    "handbag": {"name": "Handbag", "threshold": 1, "alert": True},
}

# These are the default COCO classes that YOLOv8 can detect
# We're using a subset for our retail inventory example
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Report Configuration
REPORTS_DIR = "reports"
REPORT_FREQUENCY = "daily"  # Options: "hourly", "daily" 