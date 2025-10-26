import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, start_frame=0):
    # Extract the video filename without extension
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create the output folder in the same directory as the video
    output_folder = os.path.join(os.path.dirname(video_path), video_filename)
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = 0
    with tqdm(total=total_frames - start_frame, desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as an image
            frame_name = f"{frame_number + 1:05d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

            frame_number += 1
            pbar.update(1)

    # Release the video capture object
    cap.release()
    print(f"Frames saved in folder: {output_folder}")

# Example usage
video_path = "/media/kisna/bkp_data/DeOldify/RemoveWaterMark/video-watermark-removal/meraNaamChinChinChu.mkv"  # Replace with your video file path
save_video_frames(video_path, start_frame=1)