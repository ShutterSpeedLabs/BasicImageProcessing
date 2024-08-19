import cv2
import numpy as np
import os
import re

def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    return class_idx

def get_class_name(class_idx):
    class_map = {
        1: "cars", 2: "pedestrians", 3: "trucks", 4: "smallVehicle",
        5: "utilityVehicle", 6: "bicycle", 7: "tractor"
    }
    return class_map.get(class_idx, "unknown")

def process_image_and_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    unique_values = np.unique(mask)
    instances = [value for value in unique_values if value != 0]
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(8, 3), dtype=np.uint8)
    
    yolo_annotations = []
    
    for instance in instances:
        class_idx = decode_pixel(instance)
        class_name = get_class_name(class_idx)
        
        instance_mask = (mask == instance).astype(np.uint8) * 255
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            color = tuple(map(int, colors[class_idx]))
            cv2.drawContours(img, [largest_contour], 0, color, 2)
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
    
    return img, yolo_annotations

# Folder paths
image_folder = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic/20181204_191844/camera/cam_front_center/"
mask_folder = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/20181204_191844/instance/cam_front_center/"

# Get all mask files in the folder
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
mask_files.sort()

current_index = 0

while current_index < len(mask_files):
    mask_file = mask_files[current_index]
    
    # Extract the matching part of the filename
    match = re.search(r'(\d+_instance_frontcenter_\d+\.png)', mask_file)
    if match:
        common_part = match.group(1)
        image_file = common_part.replace('instance', 'camera')
        
        img_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img, yolo_annotations = process_image_and_mask(img_path, mask_path)
            
            cv2.imshow('Instance Segmentation', img)
            
            print(f"YOLOv8 annotations for {image_file}:")
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
            print(f"Couldn't find matching files for {mask_file}")
            current_index += 1
    else:
        print(f"Couldn't parse filename: {mask_file}")
        current_index += 1

cv2.destroyAllWindows()