import cv2
import numpy as np
import os
import re
import yaml
from pathlib import Path

# ... (keep the previous helper functions)

# Load YAML file
yaml_path = "/media/parashuram/AutoData2/a2d2_inst_seg/dataset_seg.yaml"
with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

print("YAML data structure:")
for key, value in yaml_data.items():
    print(f"{key}: {value}")

# Base directory (assuming the YAML file is in the same directory as the base of your dataset)
base_dir = Path(yaml_path).parent

# Extract image paths from YAML
train_images_dir = base_dir / yaml_data['train']
val_images_dir = base_dir / yaml_data['val']

# Function to get all image files in a directory
def get_image_files(directory):
    return [f for f in directory.rglob('*.jpg')]

train_images = get_image_files(train_images_dir)
val_images = get_image_files(val_images_dir)
all_images = train_images + val_images

print(f"\nTotal images found: {len(all_images)}")
print(f"Sample image paths: {[str(img) for img in all_images[:5]]}")

# Create a dictionary for faster lookup
image_dict = {img.name: img for img in all_images}

print(f"\nSample image names: {list(image_dict.keys())[:5]}")

# Mask folder path
mask_folder = Path("/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/")

# Find all mask files
mask_files = list(mask_folder.rglob("**/cam_front_center/*.png"))
mask_files.sort()

print(f"\nTotal mask files found: {len(mask_files)}")
print(f"Sample mask names: {[mask.name for mask in mask_files[:5]]}")

current_index = 0

while current_index < len(mask_files):
    mask_file = mask_files[current_index]
    
    # Extract the matching part of the filename
    match = re.search(r'(\d+)_instance_frontcenter_(\d+)\.png', str(mask_file.name))
    if match:
        date_part, frame_part = match.groups()
        image_file_pattern = f"{date_part}_camera_frontcenter_{frame_part}.jpg"
        
        # Try to find the matching image
        matching_image = image_dict.get(image_file_pattern)
        
        if matching_image:
            print(f"Found matching image: {matching_image}")
            img, yolo_annotations = process_image_and_mask(matching_image, mask_file)
            
            cv2.imshow('Instance Segmentation', img)
            
            print(f"YOLOv8 annotations for {matching_image.name}:")
            for annotation in yolo_annotations:
                print(' '.join(map(str, annotation)))
            print()
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):  # Spacebar
                current_index += 1
            elif key == ord('q'):  # Q key
                break
            elif key == ord('b') and current_index > 0:  # B key (optional, for going back)
                current_index -= 1
        else:
            print(f"Couldn't find matching image for {mask_file.name}")
            print(f"Looking for: {image_file_pattern}")
            # Print some nearby file names for debugging
            nearby_files = [name for name in image_dict.keys() if name.startswith(date_part)][:5]
            print(f"Some nearby file names: {nearby_files}")
            current_index += 1
    else:
        print(f"Couldn't parse filename: {mask_file.name}")
        current_index += 1

cv2.destroyAllWindows()