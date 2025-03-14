import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def calculate_frame_difference(frame1, frame2):
    """Calculate the difference between two frames using Mean Absolute Difference."""
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff)

def extract_keyframes(input_folder, output_folder, threshold=30):
    """Extract keyframes from a sequence of images in a folder."""
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No image files found in the input folder")
        return
    
    # Read first frame
    prev_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    # Save first frame as keyframe
    cv2.imwrite(os.path.join(output_folder, f"keyframe_0_{image_files[0]}"), prev_frame)
    keyframe_count = 1
    
    # Add progress bar
    for i, image_file in enumerate(tqdm(image_files[1:], desc="Extracting keyframes", unit="frame"), 1):
        current_frame = cv2.imread(os.path.join(input_folder, image_file))
        
        # Calculate difference
        diff = calculate_frame_difference(prev_frame, current_frame)
        
        # If difference is greater than threshold, save as keyframe
        if diff > threshold:
            output_path = os.path.join(output_folder, f"keyframe_{keyframe_count}_{image_file}")
            cv2.imwrite(output_path, current_frame)
            keyframe_count += 1
            prev_frame = current_frame
    
    print(f"Extracted {keyframe_count} keyframes")

if __name__ == "__main__":
    # Example usage
    input_folder = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara/video_1/"
    output_folder = "/media/kisna/bkp_data/DeOldify/video_data/video_key/yeRateinYeMausamNadiKaKinara/video_1_key/"
    extract_keyframes(input_folder, output_folder, threshold=25)
