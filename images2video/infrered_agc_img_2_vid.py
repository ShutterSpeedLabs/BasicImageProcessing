import cv2
from cv2 import sort
from cv2 import HISTCMP_CHISQR_ALT
import numpy as np
import os
import natsort
from matplotlib import pyplot as plt

def check_file_exists(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    return os.path.exists(file_path)


folderPath = f'/media/kisna/data2/city/Poland/Szczecin/2019-11-15_19.45.59_done/16BitFrames/'
filename1 = 'frameIndex_0_2019-11-15_19.45.59.png'

maxThresold = 10
minThresold = 10
histMax = 65535
histMin = 0

histMaxImg = 32768

img = cv2.imread(os.path.join(folderPath, filename1), -1)
rows,cols = img.shape
newImg = np.zeros((rows, cols), dtype = "uint8")
img_org = np.zeros((rows, cols), dtype = "uint8")
maxImgValue = np.max(img)

histr = cv2.calcHist([img], [0], None, [65535], [0, 65535], accumulate=False)
histRange = len(histr) -1
histogram8bitLUT = np.zeros((histRange,1), dtype=np.uint8)
histogram8bitLUT_org = np.zeros((histRange,1), dtype=np.uint8)

print("Histgram range: ", histRange)

for iMin in range(histRange):
    if histr[iMin] > minThresold:
        histMin = iMin
        break

for iMax in range(histRange,histMin,-1):
    if histr[iMax] > maxThresold:
        histMax = iMax
        break

histMaxMinDiff = histMax - histMin
for histIndex in range(histRange):
    pixindex = int(((histIndex - histMin)*255)/histMaxMinDiff)
    pixindex_org = (histIndex/maxImgValue)*255
    if pixindex > 255:
        pixindex = 255
    elif pixindex < 0:
        pixindex = 0
    histogram8bitLUT[histIndex] = np.uint8(pixindex)
    histogram8bitLUT_org[histIndex] = np.uint8(pixindex_org)

for i in range(rows):
    for j in range(cols):
        newImg[i,j] = histogram8bitLUT[img[i,j]]
        img_org[i,j] = histogram8bitLUT_org[img[i,j]]

print(img_org[100,100])

HoriImage = np.concatenate((img_org, newImg), axis=1)
histrNew = cv2.calcHist([newImg], [0], None, [255], [0, 255], accumulate=False)


cv2.imshow('Test image',HoriImage)
# plt.plot(histogram8bitLUT_org)
# plt.show()
cv2.waitKey()
print(filename1)


folder_path = '/media/kisna/data2/city/Denmark/Copenhagen/16BitImages/output/'
suffix = '.png'
files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

for i in range(len(files)):
    file_name = f"{i}{suffix}"
    if check_file_exists(folder_path, file_name) is False:
        print(f"The file '{file_name}' does not exist in the folder.")

