import sys
import subprocess

# Setup Python environment for deployment
try:
    from distutils_placeholder import setup_environment
    setup_environment()
except ImportError:
    print("Setting up basic environment...")
    # Create required directories
    import os
    for directory in ["reports", "temp_videos"]:
        os.makedirs(directory, exist_ok=True)

# Now import the rest of the modules
import cv2
import time
import torch
import streamlit as st
import numpy as np
import pandas as pd
import os
import requests
from io import BytesIO
from datetime import datetime
from detector import ObjectDetector
from inventory import InventoryTracker
from config import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, PRODUCTS, 
    REPORT_FREQUENCY, VIDEO_PATH, CPU_OPTIMIZED,
    CPU_SKIP_FRAMES, GPU_SKIP_FRAMES
)

# Setup page config
st.set_page_config(
    page_title="Retail Inventory Tracking",
    page_icon="📦",
    layout="wide"
)

# Add diagnostic section with toggle
def show_system_diagnostics():
    with st.expander("System Diagnostics", expanded=False):
        st.write("### System Information")
        
        # Show Python and package versions
        st.code(f"""
Python: {sys.version}
OpenCV: {cv2.__version__}
NumPy: {np.__version__}
Pandas: {pd.__version__}
        """)
        
        # Check for ultralytics and torch
        try:
            import torch
            st.write(f"PyTorch: {torch.__version__}")
            st.write(f"CUDA Available: {torch.cuda.is_available()}")
        except ImportError:
            st.write("PyTorch: Not installed")
        
        try:
            import ultralytics
            st.write(f"Ultralytics: {ultralytics.__version__}")
        except ImportError:
            st.write("Ultralytics: Not installed")
        
        # Check directories
        st.write("### Directories")
        cur_dir = os.getcwd()
        st.write(f"Working directory: {cur_dir}")
        
        # List key directories
        for directory in ["reports", "temp_videos"]:
            if os.path.exists(directory):
                st.write(f"✅ {directory}/")
            else:
                st.write(f"❌ {directory}/ (missing)")
        
        # Check for model files
        st.write("### Model Files")
        # Check current directory
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if model_files:
            for model in model_files:
                size_mb = os.path.getsize(model) / (1024*1024)
                st.write(f"✅ {model} ({size_mb:.1f} MB)")
        else:
            st.write("❌ No model files found in current directory")
        
        # Also check home cache
        try:
            cache_dir = os.path.expanduser("~/.cache")
            if os.path.exists(cache_dir):
                import glob
                cache_models = list(glob.glob(f"{cache_dir}/**/*yolo*.pt", recursive=True))
                if cache_models:
                    st.write("Models in cache:")
                    for cm in cache_models:
                        size_mb = os.path.getsize(cm) / (1024*1024)
                        st.write(f"✅ {os.path.basename(cm)} ({size_mb:.1f} MB)")
        except Exception as e:
            st.write(f"Error checking cache: {e}")
            
        # Add retry button
        if st.button("Force Download Model"):
            try:
                if os.path.exists("yolov8n.pt"):
                    os.remove("yolov8n.pt")
                    st.write("Removed existing model file")
                
                from detector import download_model
                success = download_model("yolov8n.pt")
                if success:
                    st.success("Model downloaded successfully!")
                else:
                    st.error("Failed to download model")
            except Exception as e:
                st.error(f"Error during download: {e}")

# Initialize detector and tracker
@st.cache_resource
def get_detector():
    return ObjectDetector()

@st.cache_resource
def get_tracker():
    return InventoryTracker()

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
        response = requests.get(video_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Global variables
last_report_time = datetime.now()

# UI Components
def main():
    # Title
    st.title("📦 Retail Inventory Tracking System")
    st.markdown("Using YOLOv8 to detect and track retail inventory")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Input Source")
        
        # Add diagnostics section at the top
        show_system_diagnostics()
        
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
                    with open(video_path, "wb") as f:
                        f.write(video_data.getvalue())
                    video_source = video_path
                    st.success("Sample video loaded successfully")
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
                video_path = os.path.join("temp_videos", uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                video_source = video_path
                st.success(f"Video uploaded successfully: {uploaded_file.name}")
                loop_video = st.checkbox("Loop video", value=True)
            else:
                video_source = None
                loop_video = False
        
        # Manual report generation
        st.divider()
        if st.button("Generate Inventory Report"):
            report_path = tracker.save_report()
            if report_path:
                st.success(f"Report saved to {report_path}")
            else:
                st.warning("No data available to generate report")
        
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
    
    # Load the model and tracker
    try:
        detector = get_detector()
        tracker = get_tracker()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try refreshing the page if the model fails to load.")
        return
    
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
    
    # Create progress bar and control panel
    st.subheader("Processing Controls")
    col1, col2 = st.columns(2)
    with col1:
        stop_button = st.button("Stop Processing", use_container_width=True)
    with col2:
        # Add resolution scaling to reduce processing load
        processing_quality = st.select_slider(
            "Processing Quality",
            options=["Low (Faster)", "Medium", "High (Slower)"],
            value="Medium"
        )
    progress_bar = st.progress(0)
    
    # Check if we're running on CPU and should use optimizations
    using_cpu = not hasattr(torch, 'cuda') or not torch.cuda.is_available()
    
    # Determine frame processing settings based on quality and device
    frame_quality_map = {
        "Low (Faster)": {"scale": 0.5, "skip_frames": 3 if using_cpu else 1},
        "Medium": {"scale": 0.75, "skip_frames": 2 if using_cpu else 0},
        "High (Slower)": {"scale": 1.0, "skip_frames": 1 if using_cpu else 0}
    }
    
    # Apply CPU optimization if enabled
    if using_cpu and CPU_OPTIMIZED:
        # Override with CPU optimization settings
        frame_quality_map = {
            "Low (Faster)": {"scale": 0.4, "skip_frames": CPU_SKIP_FRAMES + 2},
            "Medium": {"scale": 0.6, "skip_frames": CPU_SKIP_FRAMES},
            "High (Slower)": {"scale": 0.8, "skip_frames": max(0, CPU_SKIP_FRAMES - 1)}
        }
        st.warning("⚠️ Running in CPU-optimized mode - performance may be reduced")
    elif using_cpu:
        st.warning("⚠️ Running on CPU - processing may be slow")
    
    scale_factor = frame_quality_map[processing_quality]["scale"]
    skip_frames = frame_quality_map[processing_quality]["skip_frames"]
    
    device_info = "CPU" if using_cpu else "GPU"
    st.info(f"Processing on {device_info} at {int(scale_factor*100)}% resolution, skipping {skip_frames} frames between detections")
    
    # Get total frames for video (not applicable for webcam)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "video" else 0
    frame_count = 0
    frames_since_detection = 0
    last_processed_frame = None
    
    # Add time tracking
    start_time = time.time()
    frames_processed = 0
    detection_times = []
    
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
        
        # Only perform detection on some frames to improve performance
        if frames_since_detection >= skip_frames:
            # Resize frame for faster processing if needed
            detection_start = time.time()
            
            if scale_factor < 1.0:
                h, w = frame_rgb.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                small_frame = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Run detection on smaller frame
                processed_small_frame, detections = detector.detect(small_frame)
                
                # Resize processed frame back to original size for display
                processed_frame = cv2.resize(processed_small_frame, (w, h))
            else:
                # Run detection at full resolution
                processed_frame, detections = detector.detect(frame_rgb)
            
            detection_time = time.time() - detection_start
            detection_times.append(detection_time)
            
            # Update inventory
            tracker.update(detections)
            
            # Reset counter
            frames_since_detection = 0
            last_processed_frame = processed_frame
            frames_processed += 1
            
            # Show FPS stats every 10 frames
            if frames_processed % 10 == 0 and detection_times:
                avg_detection_time = sum(detection_times) / len(detection_times)
                fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
                fps_text = f"Processing speed: {fps:.1f} FPS (avg {avg_detection_time*1000:.0f}ms/frame)"
                
                # Add FPS info to the frame
                cv2.putText(
                    processed_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Keep only the last 30 measurements
                detection_times = detection_times[-30:]
        else:
            # Skip detection but still display the frame
            frames_since_detection += 1
            if last_processed_frame is not None:
                # Use the last processed frame to show boxes
                processed_frame = last_processed_frame
            else:
                processed_frame = frame_rgb
        
        # Display the processed frame
        feed_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
        
        # Update progress for video files
        if source_type == "video" and total_frames > 0:
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
        
        # Update inventory display (less frequently to reduce UI updates)
        if frame_count % 10 == 0 or frames_since_detection == 0:
            update_inventory_display(inventory_placeholder, alerts_placeholder, tracker)
        
        # Check if we need to save a report
        now = datetime.now()
        if REPORT_FREQUENCY == "hourly" and now.hour != last_report_time.hour:
            tracker.save_report()
            last_report_time = now
        elif REPORT_FREQUENCY == "daily" and now.day != last_report_time.day:
            tracker.save_report()
            last_report_time = now
            
        # Adaptive delay based on device and processing time
        # On CPU, use longer delay to prevent UI freezing
        if using_cpu and CPU_OPTIMIZED:
            # More aggressive delay for CPU to prevent overwhelming
            time.sleep(delay * 2.0) 
        else:
            # Standard delay
            time.sleep(delay * 1.5)
        
        # Check if stop button was clicked
        if stop_button:
            break
    
    # Final stats
    total_time = time.time() - start_time
    if frames_processed > 0 and total_time > 0:
        avg_fps = frames_processed / total_time
        st.success(f"Processed {frames_processed} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    
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
    main() 
