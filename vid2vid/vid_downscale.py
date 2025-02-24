import cv2
import os
from tqdm import tqdm

def downscale_video(input_path, width, height, start_frame=0):
    # Extract the video filename without extension
    video_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create the output path in the same directory as the input file
    output_filename = f"{video_filename}_downscaled.mp4"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)
    
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with tqdm(total=total_frames - start_frame, desc="Downscaling Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (width, height))
            out.write(resized_frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_path = "/media/kisna/dataset/Project_Bollywood/yt_dl/Ankhiyon Ke Jharokhon Se-Hemlata [HD-1080p].mp4"  # Replace with your video file path
width = 640
height = 368
start_frame = 1

downscale_video(input_path, width, height, start_frame)