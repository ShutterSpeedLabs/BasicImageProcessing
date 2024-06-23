import cv2
import os
import time

# Folder containing the images
image_folder = '/media/parashuram/AutoData/nuImagesMini/sweeps/CAM_FRONT'

# Get the list of image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))])

if not image_files:
    print("No image files found in the directory.")
else:
    # Display each image
    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        
        # Read the image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Display the image
        cv2.imshow('Image', img)
        
        # Wait for a while before displaying the next image
        # Adjust the delay to control the frame rate (e.g., 30 fps -> 1000/30 = ~33 ms)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
