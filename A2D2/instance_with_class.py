filename = f'/media/parashuram/AutoData/A2D2/camera_lidar_semantic_instance/20181204_191844/instance/cam_front_center/20181204191844_instance_frontcenter_000008764.png'
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

# Function to decode class and instance ID
def decode_pixel(pixel_value):
    class_idx = pixel_value >> 10
    return class_idx

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

# Create a color image for visualization
vis_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Generate random colors for each class
np.random.seed(42)  # for reproducibility
colors = np.random.randint(0, 255, size=(8, 3), dtype=np.uint8)  # 8 classes including background

for instance in instances:
    class_idx = decode_pixel(instance)
    class_name = get_class_name(class_idx)
    
    # Create binary mask for this instance
    mask = (img == instance).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour
        color = tuple(map(int, colors[class_idx]))
        cv2.drawContours(vis_img, [largest_contour], 0, color, 2)
        
        # Add label
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(vis_img, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Instance Segmentation with Class Labels')
plt.tight_layout()
plt.show()