import os
import shutil
from tqdm import tqdm

def process_image_folders(input_base_path, output_base_path, images_per_folder=2500):
    """
    Read images from input folders and create new folders with specified number of images.
    
    Args:
        input_base_path (str): Path to the input base folder containing video_* folders
        output_base_path (str): Path where the output folders will be created
        images_per_folder (int): Number of images to store in each output folder
    """
    # Create output base directory
    os.makedirs(output_base_path, exist_ok=True)
    
    # Get all input folders sorted numerically
    input_folders = sorted([f for f in os.listdir(input_base_path) 
                          if f.startswith('video_') and 
                          os.path.isdir(os.path.join(input_base_path, f))],
                         key=lambda x: int(x.split('_')[1]))
    
    # Initialize counters
    total_image_count = 0
    output_folder_count = 1
    current_folder_image_count = 0
    
    # Create first output folder
    current_output_folder = os.path.join(output_base_path, f"video_{output_folder_count}")
    os.makedirs(current_output_folder, exist_ok=True)
    
    # Count total images for progress bar
    total_images = sum(len([f for f in os.listdir(os.path.join(input_base_path, folder))
                           if f.endswith('.png')])
                      for folder in input_folders)
    
    # Process each input folder
    with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
        for input_folder in input_folders:
            input_folder_path = os.path.join(input_base_path, input_folder)
            
            # Get all PNG files in the current input folder sorted numerically
            image_files = sorted([f for f in os.listdir(input_folder_path) 
                                if f.endswith('.png')],
                               key=lambda x: int(x.split('.')[0]))
            
            # Process each image in the current folder
            for image_file in image_files:
                input_image_path = os.path.join(input_folder_path, image_file)
                
                # Generate new image name maintaining the 5-digit format
                new_image_name = f"{current_folder_image_count + 1:05d}.png"
                output_image_path = os.path.join(current_output_folder, new_image_name)
                
                # Copy the image
                shutil.copy2(input_image_path, output_image_path)
                
                current_folder_image_count += 1
                total_image_count += 1
                pbar.update(1)
                
                # Check if current folder is full
                if current_folder_image_count >= images_per_folder:
                    output_folder_count += 1
                    current_folder_image_count = 0
                    current_output_folder = os.path.join(output_base_path, f"video_{output_folder_count}")
                    os.makedirs(current_output_folder, exist_ok=True)
    
    print(f"\nTotal images processed: {total_image_count}")
    print(f"Total output folders created: {output_folder_count}")
    print(f"Output saved in: {output_base_path}")

if __name__ == "__main__":
    # Example usage
    input_path = "/media/kisna/bkp_data/DeOldify/video_data/video_parts/yeRateinYeMausamNadiKaKinara"  # Replace with your input folder path
    output_path = "/media/kisna/bkp_data/DeOldify/video_data/video_parts/yeRateinYeMausamNadiKaKinara_sorted/"  # Replace with your output folder path
    images_per_folder = 2000  # Specify number of images per output folder
    
    try:
        process_image_folders(input_path, output_path, images_per_folder)
    except Exception as e:
        print(f"Error processing folders: {str(e)}")
