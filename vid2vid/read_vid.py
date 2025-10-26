import cv2

video_path = '/media/kisna/dataset/Project_Bollywood/siggraphasia2019_remastering/video_2.mp4'
print("Reading video:", video_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video file {video_path}")

nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties - Frame count: {nframes}, Width: {v_w}, Height: {v_h}")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame of the video")

frame_h, frame_w = frame.shape[:2]
print(f"First frame dimensions - Width: {frame_w}, Height: {frame_h}")

minwh = min(frame_w, frame_h)
if minwh == 0:
    raise ValueError("Frame width or height is zero!")

mindim = 512  # define the target minimum dimension

scale = mindim / minwh
print(f"Scale factor: {scale}")
