import os
import cv2
from cv2 import sort
import numpy as np
import natsort
from matplotlib import pyplot as plt

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


dir_list = os.listdir(folder_path)
dir_list1 = natsort.natsorted(dir_list)
def applyAGC(img):
    hist = cv2.calcHist([img], [0], None, [65535], [0, 65535], accumulate=False)
    hist = hist.flatten()
    return hist

for filename in dir_list1:
    img = cv2.imread(os.path.join(folder_path, filename), -1)
    # equ = cv2.equalizeHist(img)
    histr = cv2.calcHist([img], [0], None, [65535], [0, 65535], accumulate=False)
    histr = histr.flatten()
    cv2.imshow('Test image',img)
    plt.plot(histr)
    plt.show()
    cv2.waitKey()
    print(filename)
#     # cv2.destroyAllWindows()