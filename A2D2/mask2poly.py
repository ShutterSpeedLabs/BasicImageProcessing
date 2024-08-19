filename = f'/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/20181204_191844/instance/cam_front_center/20181204191844_instance_frontcenter_000008764.png'

import cv2
import numpy as np

# Load the image
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

# Function to decode class and instance ID
def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    instance_id = pixel_value & 0x3FF
    return class_idx, instance_id

# Function to get class name
def get_class_name(class_idx):
    class_map = {
        1: "cars", 2: "pedestrians", 3: "trucks", 4: "smallVehicle",
        5: "utilityVehicle", 6: "bicycle", 7: "tractor"
    }
    return class_map.get(class_idx, "unknown")

# Find unique instances
unique_values = np.unique(img)
instances = [value for value in unique_values if value != 0]  # Exclude background

yolo_format = []

for instance in instances:
    class_idx, instance_id = decode_pixel(instance)
    class_name = get_class_name(class_idx)
    
    # Create binary mask for this instance
    mask = (img == instance).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to relative coordinates
        height, width = img.shape
        polygon = approx.reshape(-1, 2)
        polygon = polygon.astype(float)
        polygon[:, 0] /= width
        polygon[:, 1] /= height
        
        # Flatten the polygon points
        flat_polygon = polygon.reshape(-1).tolist()
        
        # YOLOv8 format: class x1 y1 x2 y2 ... xn yn
        yolo_line = [class_idx] + flat_polygon
        yolo_format.append(yolo_line)

# Print YOLOv8 format
for line in yolo_format:
    print(' '.join(map(str, line)))