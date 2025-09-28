import os
from pathlib import Path
import cv2
from tqdm import tqdm

def create_template_images(source_dir, output_dir, reference_image_path):
    """Create template mask images for a given directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read reference image and convert to grayscale
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        raise ValueError(f"Could not read reference image: {reference_image_path}")
    
    # Convert reference image to grayscale
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Get all files from source directory
    source_files = os.listdir(source_dir)
    
    # Filter for specific extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
    source_files = [f for f in source_files if Path(f).suffix.lower() in valid_extensions]
    
    print(f"Found {len(source_files)} files in {source_dir}")
    
    # Create grayscale images with matching names
    for filename in tqdm(source_files, desc="Creating mask images"):
        name = Path(filename).stem
        new_image_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(new_image_path, ref_img_gray)

def create_mask_folders(input_path, output_base_path, reference_image_path):
    """Create folder structure and mask images"""
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist!")
        return
    
    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        print(f"Reference image {reference_image_path} does not exist!")
        return
    
    # Get the base folder name from input path
    base_folder_name = os.path.basename(os.path.normpath(input_path))
    
    # Create main output folder with same name as input base folder
    main_output_folder = os.path.join(output_base_path, base_folder_name)
    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)
        print(f"Created main output folder: {main_output_folder}")
    
    # Get all folders in the input path
    folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    
    # Process each folder
    for folder in folders:
        # Get the source folder path (where the video frames are)
        source_folder = os.path.join(input_path, folder)
        
        # Create mask folder directly in the output folder
        mask_folder = os.path.join(main_output_folder, f"{folder}_mask")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        
        print(f"\nProcessing folder: {folder}")
        print(f"Creating mask folder:")
        print(f"  - {mask_folder}")
        
        # Create template images in the mask folder
        try:
            create_template_images(source_folder, mask_folder, reference_image_path)
            print(f"Successfully created mask images for {folder}")
        except Exception as e:
            print(f"Error creating mask images for {folder}: {str(e)}")

if __name__ == "__main__":
    # Example usage with actual paths
    input_path = "/media/kisna/bkp_data/DeOldify/video_data/videos/yeRateinYeMausamNadiKaKinara"
    output_base_path = "/media/kisna/bkp_data/DeOldify/video_data/proPainterVidOut/"
    reference_image = "/media/kisna/bkp_data/DeOldify/video_data/proPainter_out/mask.png"
    
    create_mask_folders(input_path, output_base_path, reference_image)