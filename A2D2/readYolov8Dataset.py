import cv2
import numpy as np
import os
from pathlib import Path
import yaml
from screeninfo import get_monitors

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def read_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line_num, line in enumerate(lines, 1):
        try:
            data = line.strip().split()
            if len(data) < 5:  # Minimum: class_id + 2 points (4 coordinates)
                print(f"Warning: Line {line_num} has insufficient data: {line.strip()}")
                continue
            class_id = int(data[0])
            polygon = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
            annotations.append((class_id, polygon))
        except ValueError as e:
            print(f"Error parsing line {line_num}: {line.strip()}")
            print(f"Error details: {str(e)}")
    
    return annotations

def draw_annotations(image, annotations, class_names):
    h, w = image.shape[:2]
    for class_id, polygon in annotations:
        polygon = (polygon * np.array([w, h])).astype(int)
        cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        cv2.putText(image, label, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def display_dataset(yaml_path):
    config = read_yaml(yaml_path)
    
    dataset_path = Path(config.get('path', '/media/parashuram/AutoData2/a2d2_instance/'))
    class_names = config.get('names', [])
    
    if not class_names:
        print("Warning: No class names found in YAML file.")
    else:
        print("Class names:", class_names)
    
    image_dir = dataset_path / "images" / "train"
    label_dir = dataset_path / "labels" / "train"
    
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return
    if not label_dir.exists():
        print(f"Label directory not found: {label_dir}")
        return
    
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))
    total_images = len(image_files)
    images_with_labels = 0
    
    monitor = get_monitors()[0]
    max_width = int(monitor.width)
    max_height = int(monitor.height* 0.9)
    # max_width = 1920
    # max_height = 1080
    
    for i, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        
        label_path = label_dir / (img_path.stem + ".txt")
        print(f"Checking for label file: {label_path}")
        
        if label_path.exists():
            print(f"Label file found: {label_path}")
            try:
                annotations = read_yolo_labels(str(label_path))
                if annotations:
                    image_with_annotations = draw_annotations(image.copy(), annotations, class_names)
                    images_with_labels += 1
                else:
                    print(f"No valid annotations found in {label_path}")
                    image_with_annotations = image.copy()
                    cv2.putText(image_with_annotations, "No valid annotations", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error reading label file {label_path}: {str(e)}")
                image_with_annotations = image.copy()
                cv2.putText(image_with_annotations, "Error reading label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print(f"No label file found for {img_path.name}")
            image_with_annotations = image.copy()
            cv2.putText(image_with_annotations, "No label file", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        resized_image = resize_image(image_with_annotations, max_width, max_height)
        
        cv2.imshow("YOLOv8 Segmentation Annotation", resized_image)
        cv2.setWindowTitle("YOLOv8 Segmentation Annotation", f"Image {i}/{total_images}")
        
        key = cv2.waitKey(0)
        if key == ord(' '):
            continue
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"Total images: {total_images}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Images without labels: {total_images - images_with_labels}")

# Usage
yaml_path = "/media/parashuram/AutoData2/a2d2_instance/dataset.yaml"
display_dataset(yaml_path)