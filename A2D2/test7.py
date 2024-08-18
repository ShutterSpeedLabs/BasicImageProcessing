import cv2
import numpy as np
import os
import re
import shutil

def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    return class_idx

def process_image_and_mask(img_path, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    unique_values = np.unique(mask)
    instances = [value for value in unique_values if value != 0]
    
    yolo_annotations = []
    
    for instance in instances:
        class_idx = decode_pixel(instance)
        
        instance_mask = (mask == instance).astype(np.uint8) * 255
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convert to YOLOv8 format
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            height, width = mask.shape[:2]
            polygon = approx.reshape(-1, 2)
            polygon = polygon.astype(float)
            polygon[:, 0] /= width
            polygon[:, 1] /= height
            
            flat_polygon = polygon.reshape(-1).tolist()
            yolo_line = [class_idx - 1] + flat_polygon  # YOLOv8 uses 0-indexed classes
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations

# Folder paths
image_folder = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic/20181204_191844/camera/cam_front_center/"
mask_folder = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/20181204_191844/instance/cam_front_center/"
output_folder = "/media/parashuram/AutoData2/video_out/"

# Create output folders if they don't exist
labels_folder = os.path.join(output_folder, "labels")
images_folder = os.path.join(output_folder, "images")
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Get all mask files in the folder
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

for mask_file in mask_files:
    # Extract the matching part of the filename
    match = re.search(r'(\d+_instance_frontcenter_\d+\.png)', mask_file)
    if match:
        common_part = match.group(1)
        image_file = common_part.replace('instance', 'camera')
        
        img_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            yolo_annotations = process_image_and_mask(img_path, mask_path)
            
            # Generate label file name
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)
            
            # Write YOLO annotations to file
            with open(label_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(' '.join(map(str, annotation)) + '\n')
            
            # Copy and convert the image file to JPG
            output_image_file = os.path.splitext(image_file)[0] + '.jpg'
            output_image_path = os.path.join(images_folder, output_image_file)
            
            # Read the image, convert to JPG, and save
            img = cv2.imread(img_path)
            cv2.imwrite(output_image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            print(f"Generated label file: {label_file}")
            print(f"Copied and converted image file: {output_image_file}")
        else:
            print(f"Couldn't find matching files for {mask_file}")
    else:
        print(f"Couldn't parse filename: {mask_file}")