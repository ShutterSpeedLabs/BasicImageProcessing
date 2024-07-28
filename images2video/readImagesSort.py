import os

def print_files_in_folder_sorted(folder_path):
    # Get the list of files in the specified folder
    files = os.listdir(folder_path)
    
    # Sort files numerically
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    
    # Iterate through the sorted list of files and print each file name
    for file in sorted_files:
        print(file)
    
    # Print the total number of files
    print("Total Number of Files:", len(files))

# Replace '/path/to/your/folder' with the actual path to the folder you want to explore
folder_path = '/media/parashuram/AutoData2/city/Denmark/Copenhagen/16BitImages/output/'
print_files_in_folder_sorted(folder_path)
