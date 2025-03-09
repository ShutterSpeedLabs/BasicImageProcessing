from scenedetect import detect, ContentDetector
import cv2
import os

def extract_keyframes(video_path, threshold=27.0, min_scene_len=15):
    # Create output directory with same name as video file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), video_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Detect scenes in video
    try:
        print("Detecting scenes...")
        scenes = detect(video_path, ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        print(f"Found {len(scenes)} scenes")
        
        if not scenes:
            print("No scenes detected. Falling back to regular interval extraction...")
            return extract_frames_interval(video_path, output_dir, interval_sec=5)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file")
        
        frames_extracted = 0
        for i, scene in enumerate(scenes):
            # Get middle frame of the scene
            middle_frame = (scene[0].frame_num + scene[1].frame_num) // 2
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                output_path = os.path.join(output_dir, f'{video_name}_keyframe_{i:04d}.jpg')
                cv2.imwrite(output_path, frame)
                frames_extracted += 1
                print(f"Extracted frame {frames_extracted} at position {middle_frame}")
        
        cap.release()
        return frames_extracted
    
    except Exception as e:
        print(f"Error during scene detection: {str(e)}")
        return 0

def extract_frames_interval(video_path, output_dir, interval_sec=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * interval_sec)
    
    frames_extracted = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for frame_no in range(0, total_frames, interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f'{video_name}_keyframe_{frames_extracted:04d}.jpg')
            cv2.imwrite(output_path, frame)
            frames_extracted += 1
            print(f"Extracted frame {frames_extracted} at position {frame_no}")
    
    cap.release()
    return frames_extracted

if __name__ == "__main__":
    video_path = "/media/kisna/bkp_data/DeOldify/video_data/video_out/rrtn2/rrtn7/test_results_30_rec1/LekePehlaPehlaPyar/video.mp4"
    
    num_keyframes = extract_keyframes(video_path, threshold=30.0, min_scene_len=15)
    print(f"Total frames extracted: {num_keyframes}")
