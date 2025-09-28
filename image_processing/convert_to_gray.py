import cv2
import os
from pathlib import Path
from tqdm import tqdm

def convert_to_grayscale(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if Path(f).suffix.lower() in valid_extensions]
    
    if not image_files:
        print("No valid image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Converting to grayscale and saving to: {output_dir}")
    
    # Process each image with progress bar
    for filename in tqdm(image_files, desc="Converting images"):
        # Read image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create output path
            output_path = os.path.join(output_dir, f"gray_{filename}")
            
            # Save grayscale image
            cv2.imwrite(output_path, gray)
        else:
            print(f"Failed to read: {filename}")

def main():
    # Configure your paths here
    input_dir = "/path/to/input/folder"  # Folder containing original images
    output_dir = "/path/to/output/folder"  # Where grayscale images will be saved
    
    try:
        convert_to_grayscale(input_dir, output_dir)
        print("\nConversion completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
