"""
Test PyTorch and Ultralytics compatibility with the current Python environment
"""

def test_torch():
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test tensor creation
        x = torch.rand(5, 3)
        print(f"Test tensor shape: {x.shape}")
        
        # Test device availability
        print(f"Available device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        return True
    except Exception as e:
        print(f"PyTorch test failed: {e}")
        return False

def test_ultralytics():
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
        
        # Import YOLO
        from ultralytics import YOLO
        print("YOLO class imported successfully")
        
        # Try to load a model (without downloading)
        if ultralytics.__version__.startswith('8.3') or ultralytics.__version__.startswith('8.4'):
            print("Using newer ultralytics API")
        else:
            print("Using traditional ultralytics API")
        
        return True
    except Exception as e:
        print(f"Ultralytics import error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 40)
    print("TESTING PYTORCH")
    print("=" * 40)
    test_torch()
    
    print("\n" + "=" * 40)
    print("TESTING ULTRALYTICS")
    print("=" * 40)
    test_ultralytics()
    
    print("\n" + "=" * 40)
    print("TESTING MODEL DIRECTORIES")
    print("=" * 40)
    import os
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if .pt file exists
    model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if model_files:
        print(f"Found model files: {model_files}")
    else:
        print("No model files found in current directory")
        
    # Check for common directories
    for directory in ["reports", "temp_videos"]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}") 