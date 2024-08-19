import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    return class_idx

def get_class_name(class_idx):
    class_map = {
        1: "cars", 2: "pedestrians", 3: "trucks", 4: "smallVehicle",
        5: "utilityVehicle", 6: "bicycle", 7: "tractor"
    }
    return class_map.get(class_idx, "unknown")

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    unique_values = np.unique(img)
    instances = [value for value in unique_values if value != 0]
    
    vis_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(8, 3), dtype=np.uint8)
    
    for instance in instances:
        class_idx = decode_pixel(instance)
        class_name = get_class_name(class_idx)
        
        mask = (img == instance).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            color = tuple(map(int, colors[class_idx]))
            cv2.drawContours(vis_img, [largest_contour], 0, color, 2)
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(vis_img, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return vis_img

# Folder containing the mask images
folder_path = '/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/20181204_191844/instance/cam_front_center/'

# Get all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Sort the files to process them in order

current_index = 0

while current_index < len(image_files):
    img_path = os.path.join(folder_path, image_files[current_index])
    vis_img = process_image(img_path)
    
    cv2.imshow('Instance Segmentation', vis_img)
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord(' '):  # Spacebar
        current_index += 1
    elif key == ord('q'):  # Q key
        break
    elif key == ord('b') and current_index > 0:  # B key (optional, for going back)
        current_index -= 1

cv2.destroyAllWindows()