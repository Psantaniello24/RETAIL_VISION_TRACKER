"""
This file helps ensure distutils compatibility with Python 3.12+
"""

def install_distutils():
    import sys
    import subprocess
    
    try:
        import distutils
        print("Distutils is already installed!")
        return True
    except ImportError:
        print("Distutils not found, attempting to install...")
        
        try:
            subprocess.check_call([
                sys.executable,
                "-m", "pip", "install", 
                "setuptools>=59.0.0",
                "--force-reinstall"
            ])
            return True
        except Exception as e:
            print(f"Failed to install distutils: {e}")
            return False

if __name__ == "__main__":
    install_distutils() 