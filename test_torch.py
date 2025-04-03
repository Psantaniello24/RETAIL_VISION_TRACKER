"""
Test PyTorch compatibility with the current Python environment
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

if __name__ == "__main__":
    test_torch()
    
    # Also try importing ultralytics
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"Ultralytics version: {ultralytics.__version__}")
        print("YOLO import successful")
    except Exception as e:
        print(f"Ultralytics import error: {e}") 