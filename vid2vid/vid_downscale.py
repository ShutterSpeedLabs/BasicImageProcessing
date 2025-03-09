import cv2
import os
import subprocess
from tqdm import tqdm

def downscale_video(input_path, width, height, start_frame=0):
    # Extract the video filename without extension
    video_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{video_filename}_downscaled.mp4"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Use ffmpeg directly for video processing
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',  # Use H.264 codec
        '-preset', 'medium',  # Balance between speed and compression
        '-crf', '23',  # Constant Rate Factor (0-51, lower means better quality)
        '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
        output_path
    ]

    try:
        print(f"Processing video: {input_path}")
        print(f"Output will be saved as: {output_path}")
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video processing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    input_path = "/media/kisna/bkp_data/DeOldify/BnW_Videos/येह रातें येह मौसम.mp4"
    width = 640
    height = 368
    downscale_video(input_path, width, height)