#/media/parashuram/ADAS/A2D2_test/camera_lidar-20180810150607_camera_frontcenter/camera_lidar/20180810_150607/camera/cam_front_center/

import os
import cv2

# Path to the folder containing images
folder_path = '/media/parashuram/ADAS/A2D2_test/camera_lidar-20180810150607_camera_frontcenter/camera_lidar/20180810_150607/camera/cam_front_center/'  # Replace with your folder path

# Get list of all files in the folder
files = os.listdir(folder_path)

# Filter out only image files (you can add more extensions if needed)
image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Display the number of image files
print(f"Number of image files: {len(image_files)}")

# Loop through each image and display it
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image is read correctly
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue
    
    # Display the image
    cv2.imshow('Image Viewer', image)
    
    # Wait for a key press, if 'q' is pressed, exit the loop
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
