import os
from PIL import Image
from tqdm import tqdm

def convert_png_to_jpg(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    # Convert each PNG file to JPG
    for png_file in tqdm(png_files, desc=f"Converting {os.path.basename(input_folder)}"):
        png_path = os.path.join(input_folder, png_file)
        jpg_file = os.path.splitext(png_file)[0] + '.jpg'
        jpg_path = os.path.join(output_folder, jpg_file)

        # Open PNG image and convert to RGB (in case it's RGBA)
        with Image.open(png_path) as img:
            rgb_img = img.convert('RGB')
            # Save as JPG
            rgb_img.save(jpg_path, 'JPEG')

# Set up input and output directories
input_base_dir = '/media/parashuram/ADAS/yolov8_seg/images/'
output_base_dir = '/media/parashuram/AutoData2/a2d2_yolov8_seg_jpg/images/'

# Convert train folder
input_train_dir = os.path.join(input_base_dir, 'train')
output_train_dir = os.path.join(output_base_dir, 'train')
convert_png_to_jpg(input_train_dir, output_train_dir)

# Convert val folder
input_val_dir = os.path.join(input_base_dir, 'val')
output_val_dir = os.path.join(output_base_dir, 'val')
convert_png_to_jpg(input_val_dir, output_val_dir)

print("Conversion completed!")