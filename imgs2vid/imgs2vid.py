import cv2
import os
from tqdm import tqdm
import warnings

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps

def create_video_from_images(image_folder, output_path, fps, audio_path=None):
    if audio_path:
        warnings.warn("Audio cannot be added using OpenCV. Audio will be ignored.")
    
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        raise Exception("No images found in the folder")

    print(f"Found {len(images)} images")
    print(f"Creating video with FPS: {fps}")

    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = first_image.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Write each image to video
        for img_name in tqdm(images, desc="Creating video"):
            frame = cv2.imread(os.path.join(image_folder, img_name))
            out.write(frame)
    finally:
        out.release()

    print(f"Video saved as: {output_path}")

def main(video_path, image_folder, output_video_path):
    # Get FPS from original video
    fps = get_video_fps(video_path)
    print(f"Warning: Audio from original video will not be included in the output")
    create_video_from_images(image_folder, output_video_path, fps=fps)

if __name__ == "__main__":
    video_path = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara.mp4"
    image_folder = "/media/kisna/bkp_data/DeOldify/vid_data_colorize/yerateinyemausamnadikakinara/video_2"
    output_video_path = "/media/kisna/bkp_data/DeOldify/vid_data_colorize/yerateinyemausamnadikakinara/video_2.mp4"
    
    main(video_path, image_folder, output_video_path)