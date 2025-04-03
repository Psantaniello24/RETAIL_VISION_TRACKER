"""
Support for Python 3.12+ environments
"""

def setup_environment():
    import sys
    import subprocess
    import os
    
    print("Setting up Python environment for deployment...")
    
    try:
        # Check and install setuptools
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install", 
            "setuptools>=67.0.0",
            "--upgrade"
        ])
        
        # Check torch version
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
        except ImportError:
            print("PyTorch not installed yet, will be installed from requirements.txt")
        
        # Create required directories
        for directory in ["reports", "temp_videos"]:
            os.makedirs(directory, exist_ok=True)
            
        return True
    except Exception as e:
        print(f"Setup warning (non-fatal): {e}")
        return False

if __name__ == "__main__":
    setup_environment() 