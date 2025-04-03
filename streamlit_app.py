import sys
import subprocess
import traceback

# Configure startup behavior for better logging
print("Starting Retail Inventory Tracking System...")
print(f"Python version: {sys.version}")

# Try to ensure distutils compatibility
try:
    import distutils
    print("Distutils is available")
except ImportError:
    print("Attempting to install distutils compatibility...")
    try:
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install", 
            "setuptools>=59.0.0",
            "--force-reinstall"
        ])
        print("Setuptools installed successfully")
    except Exception as e:
        print(f"Warning: Failed to install distutils: {e}")
        traceback.print_exc()

# Now import the rest of the modules
try:
    import cv2
    import time
    import streamlit as st
    import numpy as np
    import pandas as pd
    import os
    import requests
    from io import BytesIO
    from datetime import datetime
    print("Basic imports successful")
except ImportError as e:
    print(f"Error importing basic modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import our application modules
try:
    print("Importing application modules...")
    from detector import ObjectDetector
    from inventory import InventoryTracker
    from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, PRODUCTS, REPORT_FREQUENCY, VIDEO_PATH
    print("Application modules imported successfully")
except ImportError as e:
    print(f"Error importing application modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Setup page config
st.set_page_config(
    page_title="Retail Inventory Tracking",
    page_icon="📦",
    layout="wide"
)

# Print PyTorch info if available
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch not imported")

# Initialize detector and tracker
@st.cache_resource
def get_detector():
    try:
        print("Initializing detector...")
        detector = ObjectDetector()
        print("Detector initialized successfully")
        return detector
    except Exception as e:
        print(f"Error initializing detector: {e}")
        traceback.print_exc()
        st.error(f"Error initializing object detector: {str(e)}")
        return None

@st.cache_resource
def get_tracker():
    try:
        print("Initializing inventory tracker...")
        tracker = InventoryTracker()
        print("Inventory tracker initialized successfully")
        return tracker
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        traceback.print_exc()
        st.error(f"Error initializing inventory tracker: {str(e)}")
        return None

# Sample videos available for cloud deployment
SAMPLE_VIDEOS = {
    "bottle-detection": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4",
        "name": "Bottles on shelf"
    },
    "store-customers": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4",
        "name": "Retail store customers"
    },
    "store-entrance": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "name": "Store entrance with people"
    },
    "retail-shelf": {
        "url": "https://github.com/computervision-xray/retail-store-detector/raw/main/sample_data/video1.mp4",
        "name": "Retail store shelf with products"
    }
}

# Function to download video
@st.cache_data
def download_video(video_url):
    try:
        print(f"Downloading video from {video_url}")
        response = requests.get(video_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("Video downloaded successfully")
        return BytesIO(response.content)
    except Exception as e:
        print(f"Error downloading video: {e}")
        traceback.print_exc()
        st.error(f"Error downloading video: {e}")
        return None

# Global variables
last_report_time = datetime.now()

# Create required directories
for directory in ["reports", "temp_videos"]:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        traceback.print_exc()
        st.warning(f"Could not create {directory} directory: {e}")

# UI Components
def main():
    # Title
    st.title("📦 Retail Inventory Tracking System")
    st.markdown("Using YOLOv8 to detect and track retail inventory")
    
    # Load the model and tracker
    try:
        print("Loading detector and tracker...")
        detector = get_detector()
        tracker = get_tracker()
        
        if not detector or not tracker:
            st.error("Failed to initialize detector or tracker. Please check the logs.")
            st.info("Ensure that all dependencies are correctly installed. You may need to refresh the page.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try refreshing the page if the model fails to load.")
        print(f"Error initializing components: {e}")
        traceback.print_exc()
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Input Source")
        
        # Choose input source
        input_source = st.radio(
            "Select input source:",
            ["Sample Video", "Upload Video", "Webcam"],
            index=0
        )
        
        # Handle different input sources
        if input_source == "Webcam":
            st.warning("Webcam might not be available in the cloud deployment.")
            source_type = "webcam"
            camera_id = st.selectbox("Select camera", [0, 1, 2, 3], index=0)
            video_source = camera_id
            loop_video = False
        
        elif input_source == "Sample Video":
            source_type = "video"
            selected_sample = st.selectbox(
                "Select a sample video:",
                list(SAMPLE_VIDEOS.keys()),
                format_func=lambda x: SAMPLE_VIDEOS[x]["name"]
            )
            
            if selected_sample:
                video_url = SAMPLE_VIDEOS[selected_sample]["url"]
                st.info(f"Using sample: {SAMPLE_VIDEOS[selected_sample]['name']}")
                video_data = download_video(video_url)
                if video_data:
                    # Save to temp file
                    video_path = os.path.join("temp_videos", f"{selected_sample}.mp4")
                    try:
                        with open(video_path, "wb") as f:
                            f.write(video_data.getvalue())
                        video_source = video_path
                        st.success("Sample video loaded successfully")
                    except Exception as e:
                        st.error(f"Error saving video: {e}")
                        print(f"Error saving video: {e}")
                        traceback.print_exc()
                        video_source = None
                else:
                    video_source = None
                    st.error("Failed to load sample video")
                loop_video = st.checkbox("Loop video", value=True)
            else:
                video_source = None
                loop_video = False
        
        elif input_source == "Upload Video":
            source_type = "video"
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                # Create temp directory if it doesn't exist
                os.makedirs("temp_videos", exist_ok=True)
                
                # Save the file
                try:
                    video_path = os.path.join("temp_videos", uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    video_source = video_path
                    st.success(f"Video uploaded successfully: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error saving uploaded video: {e}")
                    print(f"Error saving uploaded video: {e}")
                    traceback.print_exc()
                    video_source = None
                loop_video = st.checkbox("Loop video", value=True)
            else:
                video_source = None
                loop_video = False
        
        # Manual report generation
        st.divider()
        if st.button("Generate Inventory Report"):
            try:
                report_path = tracker.save_report()
                if report_path:
                    st.success(f"Report saved to {report_path}")
                else:
                    st.warning("No data available to generate report")
            except Exception as e:
                st.error(f"Error generating report: {e}")
                print(f"Error generating report: {e}")
                traceback.print_exc()
        
        st.divider()
        
        # Display tracked products
        st.subheader("Tracked Products")
        product_df = pd.DataFrame([
            {
                "Product": PRODUCTS[p]["name"],
                "Low Stock Threshold": PRODUCTS[p]["threshold"]
            }
            for p in PRODUCTS if PRODUCTS[p]["alert"]
        ])
        st.dataframe(product_df, hide_index=True)
        
        # GitHub link
        st.divider()
        st.markdown("[GitHub Repository](https://github.com/YOUR_USERNAME/retail-inventory-tracker)")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display feed
        if source_type == "webcam":
            st.subheader("Webcam Feed")
        else:
            st.subheader("Video Feed")
            
        feed_placeholder = st.empty()
        
        # Current frame
        try:
            # Only process video frames if a valid source is selected
            if video_source is not None and st.button("Start Processing", use_container_width=True):
                # Create inventory and alerts placeholders in column 2
                with col2:
                    inventory_container = st.container()
                    with inventory_container:
                        st.subheader("Current Inventory")
                        inventory_placeholder = st.empty()
                        
                    alerts_container = st.container()
                    with alerts_container:
                        st.subheader("Low Stock Alerts")
                        alerts_placeholder = st.empty()
                
                process_video_feed(
                    video_source, 
                    feed_placeholder, 
                    source_type, 
                    loop_video, 
                    inventory_placeholder,
                    alerts_placeholder,
                    detector,
                    tracker
                )
            else:
                feed_placeholder.info("Press 'Start Processing' to begin")
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    with col2:
        # Placeholders for inventory and alerts that will be updated
        st.subheader("Current Inventory")
        st.info("Inventory data will appear when detection starts")
        
        st.subheader("Low Stock Alerts")
        st.info("No alerts at the moment")

def process_video_feed(source, feed_placeholder, source_type, loop_video=False, 
                       inventory_placeholder=None, alerts_placeholder=None,
                       detector=None, tracker=None):
    """Process video feed without using threads"""
    global last_report_time
    
    # Initialize video capture
    if source_type == "webcam":
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        delay = 0.03  # ~30 FPS for webcam
    else:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03
    
    # Check if video opened successfully
    if not cap.isOpened():
        st.error(f"Error: Could not open video source {source}")
        return
    
    # Create progress bar
    progress_bar = st.progress(0)
    stop_button = st.button("Stop Processing")
    
    # Get total frames for video (not applicable for webcam)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "video" else 0
    frame_count = 0
    
    # Process frames
    while not stop_button:
        # Read frame
        ret, frame = cap.read()
        
        # If frame reading failed
        if not ret:
            if source_type == "video" and loop_video:
                # Reset video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            else:
                break
                
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        processed_frame, detections = detector.detect(frame_rgb)
        
        # Update inventory
        tracker.update(detections)
        
        # Display the processed frame
        feed_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
        
        # Update progress for video files
        if source_type == "video" and total_frames > 0:
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
        
        # Update inventory display
        update_inventory_display(inventory_placeholder, alerts_placeholder, tracker)
        
        # Check if we need to save a report
        now = datetime.now()
        if REPORT_FREQUENCY == "hourly" and now.hour != last_report_time.hour:
            tracker.save_report()
            last_report_time = now
        elif REPORT_FREQUENCY == "daily" and now.day != last_report_time.day:
            tracker.save_report()
            last_report_time = now
            
        # Add a small delay to control frame rate and not overwhelm the CPU
        time.sleep(delay)
        
        # Check if stop button was clicked
        if stop_button:
            break
    
    # Release resources
    cap.release()
    st.info("Processing stopped")
    progress_bar.empty()

def update_inventory_display(inventory_placeholder, alerts_placeholder, tracker):
    """Update the inventory and alerts display"""
    # Get inventory data
    if tracker.inventory and inventory_placeholder is not None:
        inventory_data = []
        for product, count in tracker.get_inventory().items():
            product_key = next((p for p in PRODUCTS if PRODUCTS[p]["name"] == product), None)
            if product_key and PRODUCTS[product_key]["alert"]:
                threshold = PRODUCTS[product_key]["threshold"]
                status = "🔴 Low" if count < threshold else "🟢 OK"
                inventory_data.append({"Product": product, "Count": count, "Status": status})
            else:
                inventory_data.append({"Product": product, "Count": count, "Status": "⚪ N/A"})
                
        inventory_df = pd.DataFrame(inventory_data)
        inventory_placeholder.dataframe(inventory_df, hide_index=True, use_container_width=True)
    
    # Get alerts data
    if alerts_placeholder is not None:
        if tracker.alerts:
            alert_data = []
            for alert in tracker.alerts:
                alert_data.append({
                    "Product": alert["product"],
                    "Current Count": alert["count"],
                    "Threshold": alert["threshold"],
                    "Time": alert["timestamp"].strftime("%H:%M:%S")
                })
            
            alert_df = pd.DataFrame(alert_data)
            alerts_placeholder.dataframe(alert_df, hide_index=True, use_container_width=True)
        else:
            alerts_placeholder.info("No alerts at the moment")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Application error: {e}")
        traceback.print_exc() 