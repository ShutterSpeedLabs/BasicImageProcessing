import cv2
import numpy as np
import argparse
import sys
import os

# Global variables for cropping
cropping = False
x_start, y_start = 0, 0
x_end, y_end = 0, 0
current_frame = None
output_dir = None  # New global variable

def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, current_frame, output_dir

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        x_end, y_end = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y
            # Draw rectangle on copy of frame
            frame_copy = current_frame.copy()
            cv2.rectangle(frame_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow('Video', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        # Create mask from selection
        if x_end - x_start > 0 and y_end - y_start > 0:
            roi = current_frame[min(y_start, y_end):max(y_start, y_end),
                              min(x_start, x_end):max(x_start, x_end)]
            # Create binary mask
            mask = np.zeros_like(current_frame)
            mask[min(y_start, y_end):max(y_start, y_end),
                 min(x_start, x_end):max(x_start, x_end)] = 255
            
            # Save files in video's directory
            mask_path = os.path.join(output_dir, 'mask.png')
            crop_path = os.path.join(output_dir, 'cropped_region.png')
            cv2.imwrite(crop_path, roi)
            cv2.imwrite(mask_path, mask)
            print(f"Saved files in: {output_dir}")
            print(f"Mask: {mask_path}")
            print(f"Cropped region: {crop_path}")

def main():
    global current_frame, output_dir
    
    # Default video path - modify this to your video path
    default_video_path = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara.mp4"
    
    # Optional argument parser
    parser = argparse.ArgumentParser(description='Create mask from video frame')
    parser.add_argument('--video_path', type=str, help='Path to the video file (optional)')
    args = parser.parse_args()
    
    # Use command line argument if provided, otherwise use default path
    video_path = args.video_path if args.video_path else default_video_path
    
    # Verify file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)
    
    # Set output directory to video's location
    output_dir = os.path.dirname(video_path)
    print(f"Output directory: {output_dir}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        sys.exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_crop)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame.copy()
            cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Space to pause/unpause
            paused = not paused
            print("Video", "paused" if paused else "resumed")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
