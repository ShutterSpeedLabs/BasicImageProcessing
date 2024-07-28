import os
import numpy as np
from skimage import io, exposure, img_as_ubyte, transform
import cv2
from tqdm import tqdm

def split_string_between_symbols(input_string, start_symbol, end_symbol):
    start_index = input_string.find(start_symbol)
    end_index = input_string.find(end_symbol, start_index + 1)
    if start_index != -1 and end_index != -1:
        return input_string[start_index + len(start_symbol):end_index]
    return None

def process_image(image_path):
    image = io.imread(image_path, as_gray=True)
    image_normalized = exposure.rescale_intensity(image, out_range=(0, 1))
    image_zoomed = transform.rescale(image_normalized, scale=2, order=3, mode='reflect', anti_aliasing=True)
    #image_agc = exposure.equalize_adapthist(image_zoomed, clip_limit=0.03)
    image_8bit = img_as_ubyte(image_zoomed)
    image_rgb = np.stack([image_8bit] * 3, axis=-1)
    return image_rgb

def create_video_from_folder(input_folder, output_video_path, fps=30):
    files = os.listdir(input_folder)
    file_dict = {}

    for filename in files:
        key = split_string_between_symbols(filename, '_', '_')
        if key is not None:
            file_dict[int(key)] = filename

    sorted_keys = sorted(file_dict.keys())

    first_image_path = os.path.join(input_folder, file_dict[sorted_keys[0]])
    first_image = process_image(first_image_path)
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Processing images and creating video...")
    for key in tqdm(sorted_keys, desc="Processing frames", unit="frame"):
        image_path = os.path.join(input_folder, file_dict[key])
        processed_image = process_image(image_path)
        video.write(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()

def main():
    input_folder = '/media/parashuram/AutoData2/city/Germany/Hamburg/2019-12-01_13.30.01_done/16BitFrames/'
    output_video_path = 'output_video.mp4'

    create_video_from_folder(input_folder, output_video_path)
    
    print(f"Video created: {output_video_path}")

if __name__ == "__main__":
    main()