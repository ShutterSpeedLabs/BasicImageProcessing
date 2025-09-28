import os
from pathlib import Path
import cv2
from tqdm import tqdm

def create_template_images(source_dir, output_dir, reference_image_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read reference image and convert to grayscale
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        raise ValueError(f"Could not read reference image: {reference_image_path}")
    
    # Convert reference image to grayscale
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Get all files from source directory
    source_files = os.listdir(source_dir)
    
    # Filter for specific extensions if needed
    valid_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
    source_files = [f for f in source_files if Path(f).suffix.lower() in valid_extensions]
    
    print(f"Found {len(source_files)} files in source directory")
    print(f"Output directory: {output_dir}")
    
    # Create grayscale images with matching names using progress bar
    for filename in tqdm(source_files, desc="Creating grayscale images"):
        name = Path(filename).stem
        new_image_path = os.path.join(output_dir, f"{name}.png")
        # Save as grayscale
        cv2.imwrite(new_image_path, ref_img_gray)

def main():
    # Configure your paths here
    source_dir = "/media/kisna/bkp_data/DeOldify/video_data/video_out/rrtn/test_results_30_rec2/test_data_sample/video_1/"  # Folder containing original files
    output_dir = "/media/kisna/bkp_data/DeOldify/video_data/proPainter_out/video_1_mask/"  # Where new images will be saved
    reference_image = "/media/kisna/bkp_data/DeOldify/video_data/proPainter_out/mask.png"  # Your template image
    
    try:
        create_template_images(source_dir, output_dir, reference_image)
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
