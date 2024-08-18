import cv2
import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm
import yaml
from multiprocessing import Pool, cpu_count


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

def process_dataset(image_root, mask_root, output_dir):
    for date_folder in os.listdir(mask_root):
        mask_date_path = os.path.join(mask_root, date_folder)
        image_date_path = os.path.join(image_root, date_folder)
        
        if not os.path.isdir(mask_date_path) or not os.path.isdir(image_date_path):
            continue
        
        mask_instance_path = os.path.join(mask_date_path, "instance")
        image_camera_path = os.path.join(image_date_path, "camera")
        
        for camera_folder in os.listdir(mask_instance_path):
            mask_camera_path = os.path.join(mask_instance_path, camera_folder)
            image_camera_path = os.path.join(image_camera_path, camera_folder)
            
            if not os.path.isdir(mask_camera_path) or not os.path.isdir(image_camera_path):
                continue
            
            for mask_file in os.listdir(mask_camera_path):
                if not mask_file.endswith('.png'):
                    continue
                
                match = re.search(r'(\d+_instance_\w+_\d+\.png)', mask_file)
                if match:
                    common_part = match.group(1)
                    image_file = common_part.replace('instance', 'camera')
                    
                    img_path = os.path.join(image_camera_path, image_file)
                    mask_path = os.path.join(mask_camera_path, mask_file)
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        yolo_annotations = process_image_and_mask(img_path, mask_path)
                        
                        # Create output directory structure
                        rel_path = os.path.relpath(image_camera_path, image_root)
                        output_image_dir = os.path.join(output_dir, "images")
                        output_label_dir = os.path.join(output_dir, "labels")
                        os.makedirs(output_image_dir, exist_ok=True)
                        os.makedirs(output_label_dir, exist_ok=True)
                        
                        # Copy image to output directory
                        #output_image_path = os.path.join(output_image_dir, image_file)
                        image = Image.open(img_path)
                        new_filename = f"{folder}_{os.path.splitext(img_path)[0]}.jpg"
                        image.convert('RGB').save(os.path.join(output_dir, "images", new_filename), 'JPEG')
                        #cv2.imwrite(output_image_path, cv2.imread(img_path))
                        
                        # Write YOLO annotations
                        output_label_path = os.path.join(output_label_dir, image_file.replace('.png', '.txt'))
                        with open(output_label_path, 'w') as f:
                            for annotation in yolo_annotations:
                                f.write(' '.join(map(str, annotation)) + '\n')
                        
                        print(f"Processed: {img_path}")
                    else:
                        print(f"Couldn't find matching files for {mask_file}")
                else:
                    print(f"Couldn't parse filename: {mask_file}")

# Set the root folders and output directory
image_root = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic"
mask_root = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance"
output_dir = "/media/parashuram/AutoData2/dataset_out/"  # Replace with your desired output path

# Process the entire dataset
process_dataset(image_root, mask_root, output_dir)