import os
import re
from pathlib import Path

def rename_frames(folder_path):
    # Get all image files recursively
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    # Sort files to maintain sequence
    image_files.sort()
    
    # First pass: rename all files to temporary names to avoid conflicts
    temp_files = []
    for i, old_path in enumerate(image_files):
        directory = os.path.dirname(old_path)
        extension = os.path.splitext(old_path)[1].lower()
        temp_path = os.path.join(directory, f"temp_{i:05d}{extension}")
        os.rename(old_path, temp_path)
        temp_files.append((temp_path, extension, directory))
    
    # Second pass: rename from temp to final names
    for i, (temp_path, extension, directory) in enumerate(temp_files, 1):  # Start from 1
        new_name = f"{i:05d}{extension}"  # 5 digits with leading zeros
        new_path = os.path.join(directory, new_name)
        os.rename(temp_path, new_path)
        print(f"Renamed: {os.path.basename(temp_path)} -> {new_name}")

if __name__ == "__main__":
    folder_path = "/media/kisna/bkp_data/DeOldify/video_data/proPainterVidOut/yeRateinYeMausamNadiKaKinara_out_1/"
    if os.path.exists(folder_path):
        rename_frames(folder_path)
        print("Renaming completed!")
    else:
        print("Invalid folder path. Please provide a valid path.")