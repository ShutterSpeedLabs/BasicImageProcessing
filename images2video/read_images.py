import os

def print_files_in_folder(folder_path):
    # Get the list of files in the specified folder
    files = os.listdir(folder_path)
    i =0
    # Iterate through the list of files and print each file name
    for file in files:
        print(file)
        i=i+1
    print("Total Number of Files: ", i)
# Replace '/path/to/your/folder' with the actual path to the folder you want to explore
folder_path = '/media/kisna/data2/city/Denmark/Copenhagen/16BitImages/output/'
print_files_in_folder(folder_path)