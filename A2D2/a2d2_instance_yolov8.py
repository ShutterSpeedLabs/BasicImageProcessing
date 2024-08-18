import cv2
import numpy as np
import os
import re
from tqdm import tqdm
import yaml
import multiprocessing

def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    return class_idx

def process_image_and_mask(img_path, mask_path, labels_folder, images_folder):
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
    
    # Generate label file name
    label_file = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
    label_path = os.path.join(labels_folder, label_file)
    
    # Write YOLO annotations to file
    with open(label_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(' '.join(map(str, annotation)) + '\n')
    
    # Copy and convert the image file to JPG
    output_image_file = os.path.splitext(os.path.basename(img_path))[0] + '.jpg'
    output_image_path = os.path.join(images_folder, output_image_file)
    
    # Read the image, convert to JPG, and save
    img = cv2.imread(img_path)
    cv2.imwrite(output_image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return label_file, output_image_file

# Root folder paths
image_root = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic/"
mask_root = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/"
output_root = "/media/parashuram/AutoData2/video_out/"

# Create output folders if they don't exist
labels_folder = os.path.join(output_root, "labels")
images_folder = os.path.join(output_root, "images")
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Function to create YOLOv8 YAML file
def create_yaml_file():
    # A2D2 dataset classes (you may need to adjust this list based on your specific dataset)
    classes = [
        "car", "truck", "pedestrian", "bicycle", "traffic_light", "traffic_sign",
        "utility_vehicle", "sidebars", "speed_bumper", "curbstone", "solid_line",
        "irrelevant"  # Add or remove classes as needed
    ]
    
    yaml_content = {
        'train': os.path.join(output_root, 'images'),  # path to training images
        'val': os.path.join(output_root, 'images'),    # path to validation images (you might want to split this)
        'nc': len(classes),  # number of classes
        'names': classes     # class names
    }
    
    yaml_path = os.path.join(output_root, 'a2d2_dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created YOLOv8 YAML file: {yaml_path}")

def process_file(mask_path):
    # Extract the matching part of the filename
    match = re.search(r'(\d+_instance_frontcenter_\d+\.png)', mask_path)
    if match:
        common_part = match.group(1)
        image_file = common_part.replace('instance', 'camera')
        
        relative_path = os.path.relpath(mask_path, mask_root)
        image_path = os.path.join(image_root, relative_path.replace('instance', 'camera'))
        
        if os.path.exists(image_path):
            return process_image_and_mask(image_path, mask_path, labels_folder, images_folder)
    return None, None

# Function to process all folders
def process_all_folders():
    all_mask_files = []
    for root, _, files in os.walk(mask_root):
        for file in files:
            if file.endswith('.png'):
                all_mask_files.append(os.path.join(root, file))
    
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpus)
    
    results = list(tqdm(pool.imap(process_file, all_mask_files), 
                        total=len(all_mask_files), 
                        desc="Processing files", 
                        unit="file"))
    
    pool.close()
    pool.join()
    
    processed_files = [result for result in results if result[0] is not None]
    print(f"Processed {len(processed_files)} files.")

# Main execution
if __name__ == '__main__':
    create_yaml_file()
    process_all_folders()
    print("Processing complete.")