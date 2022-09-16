from skimage.morphology import skeletonize
from os.path import join
import os
import re
import cv2
from tqdm import tqdm
MASKDIR='./9_8'
IMGDIR='I://chusai_release/train/images'
OUTPUT='./skeleton'
rexNum='_([0-9]+)'
fileList=os.listdir(MASKDIR)
for fileName in tqdm(fileList):
    idx=int(re.findall(rexNum,fileName)[0])
    filePath=join(MASKDIR,fileName)
    imgPath=join(IMGDIR,str(idx)+".tif")
    outputPath=join(OUTPUT,fileName)
    mask=cv2.imread(filePath)
    skeleton=skeletonize(mask)
    cv2.imwrite(outputPath,skeleton[:,:,1])