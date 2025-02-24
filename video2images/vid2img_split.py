import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, output_base_folder, width=640, height=368, frames_per_folder=2500, start_frame=0):
    # Create the base output folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = 0
    folder_count = 1
    current_folder = os.path.join(output_base_folder, f"video_{folder_count}")
    os.makedirs(current_folder, exist_ok=True)

    with tqdm(total=total_frames - start_frame, desc="Processing Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))

            # Save the frame as an image
            frame_name = f"{(frame_number % frames_per_folder) + 1:05d}.png"
            frame_path = os.path.join(current_folder, frame_name)
            cv2.imwrite(frame_path, resized_frame)

            frame_number += 1
            pbar.update(1)

            # Check if we need a new folder
            if frame_number % frames_per_folder == 0:
                folder_count += 1
                current_folder = os.path.join(output_base_folder, f"video_{folder_count}")
                os.makedirs(current_folder, exist_ok=True)

    # Release the video capture object
    cap.release()
    print(f"Frames saved in folders under: {output_base_folder}")

# Example usage
video_path = "/media/kisna/dataset/Project_Bollywood/videos/ManDoleMeraTanDole.mp4"  # Replace with your video file path
# Extract video filename without extension and use it as output folder name
video_filename = os.path.splitext(os.path.basename(video_path))[0]
output_base_folder = os.path.join(os.path.dirname(video_path), video_filename)
save_video_frames(video_path, output_base_folder, start_frame=1)
