import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from URL with progress bar
    """
    # Only create directory if filename has a directory path
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    print(f"Downloading {filename}...")
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    print(f"Download complete: {filename}")

def print_options(options):
    """Print options in a clear way"""
    print("Available sample videos:")
    for key, video in options.items():
        print(f"{key}: {video['name']}")

if __name__ == "__main__":
    # Choose one of these simpler retail store videos
    video_options = {
        "1": {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4", 
            "name": "bottles on shelf (small, 1.4MB)"
        },
        "2": {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4",
            "name": "retail store customers (medium, 2.8MB)"
        },
        "3": {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
            "name": "store entrance with people (small, 0.8MB)"
        },
        "4": {
            "url": "https://github.com/computervision-xray/retail-store-detector/raw/main/sample_data/video1.mp4",
            "name": "retail store shelf with products (medium, 3.5MB)"
        }
    }
    
    # Show options
    print_options(video_options)
    
    # Ask for choice with a default
    try:
        choice = input("\nSelect video (1-4) or press Enter for default [1]: ").strip()
        if not choice:
            choice = "1"  # Default choice
        
        if choice not in video_options:
            print(f"Invalid choice '{choice}'. Using default option 1.")
            choice = "1"
    except Exception as e:
        print(f"Input error: {e}. Using default option 1.")
        choice = "1"
    
    # Download the selected video
    selected_video = video_options[choice]
    video_url = selected_video["url"]
    
    print(f"Selected: {selected_video['name']}")
    
    # Save as demo_video.mp4
    download_file(video_url, "demo_video.mp4")
    
    print("\nVideo downloaded successfully. You can now run the app with:")
    print("streamlit run app.py")
    print("\nSelect 'Sample Video' in the input source section.") 