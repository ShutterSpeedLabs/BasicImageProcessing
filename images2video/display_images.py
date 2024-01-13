import os

def check_file_exists(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    return os.path.exists(file_path)

# Example usage:
folder_path = '/media/kisna/data2/city/Denmark/Copenhagen/16BitImages/output/'
suffix = '.png'
files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

for i in range(len(files)):
    file_name = f"{i}{suffix}"
    if check_file_exists(folder_path, file_name) is False:
        print(f"The file '{file_name}' does not exist in the folder.")


