import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import deque

class TemporalLineDetector:
    def __init__(self, history_length=5, consistency_threshold=0.6, min_line_length=30):
        self.history_length = history_length
        self.consistency_threshold = consistency_threshold
        self.min_line_length = min_line_length
        self.frame_history = deque(maxlen=history_length)
        self.mask_history = deque(maxlen=history_length)
        
    def detect_vertical_lines(self, frame):
        # Convert frame to grayscale if it's not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Store original frame in history
        self.frame_history.append(gray.copy())

        # Apply preprocessing for old black and white video
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive histogram equalization to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance vertical structures
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
        vertical_enhanced = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_vertical)
        
        # Use Sobel operator to detect vertical edges with increased sensitivity
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi

        # Create initial mask for vertical lines with relaxed constraints
        mask = np.zeros_like(gray)
        vertical_condition = (np.abs(angle) > 80) & (np.abs(angle) < 100)  # Relaxed angle constraint
        magnitude_threshold = np.percentile(magnitude_normalized, 85)  # Adaptive threshold
        magnitude_condition = magnitude_normalized > magnitude_threshold
        mask[vertical_condition & magnitude_condition] = 255

        # Apply morphological operations to connect nearby vertical lines
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        mask = cv2.dilate(mask, kernel_connect, iterations=1)
        mask = cv2.erode(mask, kernel_connect, iterations=1)
        
        # Store mask in history
        self.mask_history.append(mask)

        # Only process if we have enough history
        if len(self.mask_history) == self.history_length:
            # Create temporal consistency mask
            temporal_mask = np.zeros_like(mask)
            
            # Find lines that appear consistently across frames with relaxed threshold
            consistent_pixels = np.sum(np.stack(self.mask_history) > 0, axis=0)
            temporal_mask[consistent_pixels >= (self.consistency_threshold * self.history_length)] = 255
            
            # Remove short lines
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temporal_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_HEIGHT] < self.min_line_length:
                    temporal_mask[labels == i] = 0

            # Final clean up
            kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_CLOSE, kernel_cleanup)

        # Convert mask to 3-channel for display
        mask_3ch = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        mask_3ch[:, :, 0] = mask  # Blue channel
        mask_3ch[:, :, 1] = mask  # Green channel
        mask_3ch[:, :, 2] = mask  # Red channel

        # For debugging: add text showing detection parameters
        cv2.putText(mask_3ch, f"Detected lines: {np.count_nonzero(mask)/255}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(mask_3ch, f"Threshold: {magnitude_threshold:.1f}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        return mask_3ch

def process_video_frames(video_path, output_base_folder, width=1280, height=720, frames_per_folder=2500, 
                    start_frame=0, test_mode=False, test_duration_sec=60):
    # Create the base output folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Create separate folders for original and mask frames
    orig_base_folder = os.path.join(output_base_folder, "original")
    mask_base_folder = os.path.join(output_base_folder, "mask")
    os.makedirs(orig_base_folder, exist_ok=True)
    os.makedirs(mask_base_folder, exist_ok=True)

    # Initialize the temporal line detector
    line_detector = TemporalLineDetector(history_length=15,  # Analyze 15 frames of history
                                       consistency_threshold=0.8,  # Line must appear in 80% of frames
                                       min_line_length=50)  # Minimum vertical line length in pixels

    # Open the video file with FFMPEG backend
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # If in test mode, only process first minute
    if test_mode:
        frames_to_process = min(total_frames, fps * test_duration_sec)
        print(f"Test mode: Processing only first {test_duration_sec} seconds ({frames_to_process} frames)")
    else:
        frames_to_process = total_frames

    # Set up video writers for the side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = "test_output.mp4" if test_mode else "combined_output.mp4"
    combined_video_path = os.path.join(output_base_folder, video_name)
    out = cv2.VideoWriter(combined_video_path, fourcc, fps, (width * 2, height))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = 0
    orig_folder_count = mask_folder_count = 1
    current_orig_folder = os.path.join(orig_base_folder, f"video_{orig_folder_count}")
    current_mask_folder = os.path.join(mask_base_folder, f"video_{mask_folder_count}")
    os.makedirs(current_orig_folder, exist_ok=True)
    os.makedirs(current_mask_folder, exist_ok=True)

    with tqdm(total=frames_to_process - start_frame, desc="Processing Frames", unit="frame") as pbar:
        while frame_number < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to HD resolution
            resized_frame = cv2.resize(frame, (width, height))
            
            # Create mask for vertical lines using temporal detection
            mask = line_detector.detect_vertical_lines(resized_frame)
            
            # The mask is already in 3 channels
            mask_3ch = mask  # No conversion needed since detect_vertical_lines returns 3-channel image
            
            # Create side-by-side display
            combined = np.hstack((resized_frame, mask_3ch))
            out.write(combined)

            # Save original frame
            frame_name = f"{(frame_number % frames_per_folder) + 1:05d}.png"
            orig_frame_path = os.path.join(current_orig_folder, frame_name)
            cv2.imwrite(orig_frame_path, resized_frame)

            # Save mask frame
            mask_frame_path = os.path.join(current_mask_folder, frame_name)
            cv2.imwrite(mask_frame_path, mask)

            frame_number += 1
            pbar.update(1)

            # Check if we need new folders
            if frame_number % frames_per_folder == 0:
                orig_folder_count += 1
                mask_folder_count += 1
                current_orig_folder = os.path.join(orig_base_folder, f"video_{orig_folder_count}")
                current_mask_folder = os.path.join(mask_base_folder, f"video_{mask_folder_count}")
                os.makedirs(current_orig_folder, exist_ok=True)
                os.makedirs(current_mask_folder, exist_ok=True)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print(f"Processing complete. Output saved in: {output_base_folder}")
    print(f"Side-by-side video saved as: {combined_video_path}")

if __name__ == "__main__":
    try:
        video_path = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara.mp4"
        # Extract video filename without extension and use it as output folder name
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create test mode folder if in test mode
        test_mode = True  # Set to False for full video processing
        folder_suffix = "_test" if test_mode else "_with_masks"
        output_base_folder = os.path.join(os.path.dirname(video_path), f"{video_filename}{folder_suffix}")
        
        # Try to convert the video to H264 format first if needed
        temp_video_path = os.path.join(output_base_folder, "temp_video.mp4")
        os.makedirs(output_base_folder, exist_ok=True)
        
        if test_mode:
            # For test mode, extract only first 30 seconds of video
            os.system(f'ffmpeg -i "{video_path}" -t 30 -c:v libx264 -crf 23 "{temp_video_path}" -y')
        else:
            # Convert full video to H264 format
            os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -crf 23 "{temp_video_path}" -y')
        
        # Process the video frames
        process_video_frames(temp_video_path if os.path.exists(temp_video_path) else video_path, 
                           output_base_folder, start_frame=1, 
                           test_mode=test_mode, test_duration_sec=30)
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")