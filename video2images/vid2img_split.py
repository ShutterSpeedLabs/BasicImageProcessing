import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, output_base_folder, width=640, height=368, frames_per_folder=2500, start_frame=0):
    # Create the base output folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Open the video file with FFMPEG backend
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"  # Force H264 decoder
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

            # Save the frame as an image with maximum PNG compression for best quality
            frame_name = f"{(frame_number % frames_per_folder) + 1:05d}.png"
            frame_path = os.path.join(current_folder, frame_name)
            cv2.imwrite(frame_path, resized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

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
if __name__ == "__main__":
    try:
        video_path = "/media/kisna/bkp_data/video_color/video_org/MIle Sur Mera Tumhara - 720P High Quality - Doordarshan.mp4"  # Replace with your video file path
        # Extract video filename without extension and use it as output folder name
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_base_folder = os.path.join(os.path.dirname(video_path), video_filename)
        
        # Try to convert the video to H264 format first if needed
        temp_video_path = os.path.join(output_base_folder, "temp_video.mp4")
        os.makedirs(output_base_folder, exist_ok=True)
        
        # Convert video to H264 format
        os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -crf 23 "{temp_video_path}" -y')
        
        # Use the converted video
        save_video_frames(temp_video_path if os.path.exists(temp_video_path) else video_path, 
                         output_base_folder, frames_per_folder=2000, start_frame=1)
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
