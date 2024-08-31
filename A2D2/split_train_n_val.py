import os
import random
import shutil

# Set paths
root_dir = '/media/parashuram/AutoData2/a2d2_instace_org/'
images_dir = os.path.join(root_dir, 'images')
labels_dir = os.path.join(root_dir, 'labels')

# Create train and val subdirectories in both images and labels folders
for parent_dir in [images_dir, labels_dir]:
    os.makedirs(os.path.join(parent_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'val'), exist_ok=True)

# Set split ratio (e.g., 0.8 for 80% train, 20% val)
split_ratio = 0.9

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle the list
random.shuffle(image_files)

# Calculate split point
split_point = int(len(image_files) * split_ratio)

# Split and move files
for i, file in enumerate(image_files):
    src_img = os.path.join(images_dir, file)
    src_label = os.path.join(labels_dir, os.path.splitext(file)[0] + '.txt')
    
    if i < split_point:
        dst_img = os.path.join(images_dir, 'train', file)
        dst_label = os.path.join(labels_dir, 'train', os.path.splitext(file)[0] + '.txt')
    else:
        dst_img = os.path.join(images_dir, 'val', file)
        dst_label = os.path.join(labels_dir, 'val', os.path.splitext(file)[0] + '.txt')
    
    shutil.move(src_img, dst_img)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)

print(f"Split complete: {split_point} images in train, {len(image_files) - split_point} images in val")