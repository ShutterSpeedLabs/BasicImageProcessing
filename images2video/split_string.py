import os


def rename_files_in_folder(folder_path, old_suffix, new_suffix):
    for filename in os.listdir(folder_path):
        if filename.endswith(old_suffix):
            # Construct the new file name
            new_name = filename.replace(old_suffix, new_suffix)

            # Construct the full paths for old and new names
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} to {new_name}')


def split_string_between_symbols(input_string, start_symbol, end_symbol):
    # Find the position of the start and end symbols
    start_index = input_string.find(start_symbol)
    end_index = input_string.find(end_symbol)

    # Check if both symbols are found
    if start_index != -1 and end_index != -1:
        # Extract the substring between the symbols
        result = input_string[start_index + len(start_symbol):end_index]
        return result
    else:
        return None

# Example usage:
input_string = "This is a [sample] string."
start_symbol = "["
end_symbol = "]"

result = split_string_between_symbols(input_string, start_symbol, end_symbol)
print(result)
