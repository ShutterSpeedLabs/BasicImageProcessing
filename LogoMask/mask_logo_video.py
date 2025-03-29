import cv2
import numpy as np
from tqdm import tqdm

def select_roi(frame):
    # Select ROI
    roi = cv2.selectROI("Select Logo Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Logo Region")
    return roi

def create_mask_video(video_path, output_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Let user select logo region
    roi = select_roi(first_frame)
    x, y, w, h = map(int, roi)

    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process all frames with progress bar
    for _ in tqdm(range(total_frames), desc="Creating mask video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Create mask (white where logo should be, black elsewhere)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        # Write mask frame
        out.write(mask)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara_EDIT.mp4"
    output_video = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara_EDITmask.mp4"
    create_mask_video(input_video, output_video)
