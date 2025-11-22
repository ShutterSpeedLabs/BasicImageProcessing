import os
from pathlib import Path
from tqdm import tqdm

def rename_frames(input_path, prefix_to_remove='vi_frame_'):
    """
    Rename all images in the directory and subdirectories by removing the specified prefix
    and incrementally numbering them starting from 00001
    
    Args:
        input_path (str): Path to the directory containing images
        prefix_to_remove (str): Prefix to remove from the image names
    """
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist!")
        return
    
    # Get all subdirectories including the root
    all_dirs = set()
    for root, dirs, files in os.walk(input_path):
        all_dirs.add(root)
    
    total_renamed = 0
    
    # Process each directory separately to maintain sequential numbering
    for directory in all_dirs:
        # Get all image files in the current directory (not recursively)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) 
                and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sort files to ensure consistent ordering
        files.sort()
        
        if not files:
            continue
            
        print(f"\nProcessing directory: {directory}")
        print(f"Found {len(files)} image files")
        
        # Process each file in the current directory
        for idx, filename in enumerate(tqdm(files, desc="Renaming files"), start=1):
            if not filename.startswith(prefix_to_remove):
                continue
                
            old_path = os.path.join(directory, filename)
            # Create new filename with 5-digit sequential number
            new_filename = f"{idx:05d}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                total_renamed += 1
            except Exception as e:
                print(f"Error renaming {filename}: {str(e)}")
    
    print(f"\nTotal files renamed: {total_renamed}")
    print("Process completed!")

if __name__ == "__main__":
    # Example usage
    input_path = "/media/kisna/bkp_data/DeOldify/video_data/proPainter_out/video_2_out/"
    prefix_to_remove = "vi_frame_"
    
    # Confirm with user before proceeding
    print(f"This will rename all images in {input_path} and its subdirectories")
    print(f"by removing the prefix '{prefix_to_remove}' and adding sequential numbers")
    print("Example: 'vi_frame_00000.png' -> '00001.png'")
    confirmation = input("Do you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        rename_frames(input_path, prefix_to_remove)
    else:
        print("Operation cancelled")