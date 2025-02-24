import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, width=640, height=368, start_frame=0):
    # Get video filename without extension and create output folder with same name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(video_path), video_name)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = 1
    with tqdm(total=total_frames - start_frame, desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))

            # Save the frame as an image with 8-digit numbering
            frame_name = f"{frame_number:08d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, resized_frame)

            frame_number += 1
            pbar.update(1)

    # Release the video capture object
    cap.release()
    print(f"Frames saved in folder: {output_folder}")

# Example usage
video_path = "/media/kisna/data2/DeOldify/video_data/videos/aplamChaplam.mp4"  # Replace with your video file path
save_video_frames(video_path, start_frame=201)