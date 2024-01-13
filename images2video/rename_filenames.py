import os
from tqdm import tqdm  # Import tqdm for progress bar

def split_string_between_symbols(input_string, start_symbol, end_symbol):
    # Find the position of the start and end symbols
    start_index = input_string.find(start_symbol)
    end_index = input_string.find(end_symbol,start_index+1)

    # Check if both symbols are found
    if start_index != -1 and end_index != -1:
        # Extract the substring between the symbols
        result = input_string[start_index + len(start_symbol):end_index]
        return result
    else:
        return None


import os
import shutil
from tqdm import tqdm

def rename_and_copy_files_with_progress(source_folder, destination_folder, suffix):
    files = os.listdir(source_folder)

    # Initialize tqdm with the files list for progress bar
    with tqdm(total=len(files), desc='Processing Files', unit='file') as pbar:
        for filename in files:
            result = split_string_between_symbols(filename, start_symbol, end_symbol)
            # Construct the new file name with the specified prefix and suffix
            new_name = f"{result}{suffix}"

            # Construct the full paths for old and new names
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, new_name)

            # Rename the file
            os.rename(source_path, destination_path)

            # Copy the renamed file to the destination folder
            #shutil.copy(source_path, destination_folder)

            # Update the progress bar
            pbar.update(1)

# Example usage:
start_symbol = '_'
end_symbol = '_'
source_folder = '/media/kisna/data2/city/Denmark/Copenhagen/16BitImages/16BitFrames/'
destination_folder = '/media/kisna/data2/city/Denmark/Copenhagen/16BitImages/output/'
suffix = '.png'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

rename_and_copy_files_with_progress(source_folder, destination_folder, suffix)
