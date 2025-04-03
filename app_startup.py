#!/usr/bin/env python
"""
Startup script to prepare the environment and pre-download models
"""
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def download_yolo_model():
    """Download YOLOv8 model before starting the app"""
    model_path = "yolov8n.pt"
    
    # If model already exists, skip download
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        print(f"Model already exists at {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
        return True
    
    print(f"Downloading YOLOv8 model to {model_path}...")
    
    # Multiple URLs to try
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt",
        "https://huggingface.co/ultralytics/yolov8n/resolve/main/yolov8n.pt"
    ]
    
    # Try each URL
    for i, url in enumerate(urls):
        print(f"Attempt {i+1}/{len(urls)}: {url}")
        try:
            # Set timeout for connection
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size/1024/1024:.1f} MB")
            
            # Download with progress reporting
            downloaded = 0
            start_time = time.time()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress periodically
                        if downloaded % (1024*1024) < 8192:  # every ~1MB
                            elapsed = time.time() - start_time
                            speed = downloaded / (elapsed + 0.00001) / 1024 / 1024
                            percent = 100 * downloaded / total_size if total_size > 0 else 0
                            sys.stdout.write(f"\rDownloaded: {downloaded/1024/1024:.1f} MB ({percent:.1f}%) - {speed:.1f} MB/s")
                            sys.stdout.flush()
            
            print(f"\nDownload complete: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
            
            # Verify downloaded file
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                print("Model download successful!")
                return True
            else:
                print(f"Downloaded file too small: {os.path.getsize(model_path)} bytes")
        
        except Exception as e:
            print(f"Download failed: {e}")
    
    print("All download attempts failed")
    return False

def setup_environment():
    """Set up the environment before running the app"""
    # Create required directories
    for directory in ["reports", "temp_videos", "models"]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Try to download the model
    model_success = download_yolo_model()
    
    if not model_success:
        # Try installing ultralytics and letting it handle the download
        try:
            print("Attempting to install/upgrade ultralytics...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
            
            # Import and try to load model (which should download it)
            print("Importing ultralytics...")
            from ultralytics import YOLO
            print("Loading model (should trigger download)...")
            model = YOLO("yolov8n.pt")
            print("Model loaded successfully")
            
            # Try to find the downloaded model file
            model_cache = list(Path(os.path.expanduser("~/.cache")).rglob("*yolo*.pt"))
            if model_cache:
                print(f"Model cached at: {model_cache[0]}")
                # Copy to current directory
                import shutil
                shutil.copy(str(model_cache[0]), "yolov8n.pt")
                print(f"Copied to {os.path.abspath('yolov8n.pt')}")
            else:
                print("Could not find cached model")
        except Exception as e:
            print(f"Ultralytics approach failed: {e}")
        
    # Print system info
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Return success status
    return os.path.exists("yolov8n.pt")

if __name__ == "__main__":
    print("="*50)
    print("Starting pre-deployment setup")
    print("="*50)
    
    setup_success = setup_environment()
    
    if setup_success:
        print("\nSetup successful! Starting Streamlit app...")
        # Start the Streamlit app
        subprocess.call(["streamlit", "run", "streamlit_app.py"])
    else:
        print("\nSetup encountered issues, but will attempt to start anyway.")
        # Still try to start the app
        subprocess.call(["streamlit", "run", "streamlit_app.py"]) 