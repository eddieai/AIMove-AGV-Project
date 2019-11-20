import cv2
import glob
import os
import numpy as np
import pandas as pd
import json

keypoints = pd.read_json('keypoints.json', orient='records')

folderLocation = '/home/aimove/Desktop/Glass_Blowing_New/Blow_through_the_stick/'
foldNum = 1
frameNum = 1
depth = np.empty(0)

while True:
    # Read frames on directory
    imgPath=glob.glob(folderLocation+str(foldNum)+'D/*.png')
    if (frameNum>=len(imgPath)):
        foldNum += 1
        frameNum=1
        if not os.path.isdir(folderLocation + str(foldNum)+'D'):
            break
        imgPath=glob.glob(folderLocation+str(foldNum)+'D/*.png')

    img = cv2.imread(imgPath[frameNum-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    chunk = keypoints[np.logical_and(keypoints['Sub folder No.']==foldNum, keypoints['Frame No.']==frameNum)]
    chunk_y = np.where(np.array(chunk['Y']).astype(int)>479, 479, np.array(chunk['Y']).astype(int))
    chunk_x = np.where(np.array(chunk['X']).astype(int)>639, 639, np.array(chunk['X']).astype(int))
    chunk_depth = img[chunk_y, chunk_x]
    depth = np.concatenate((depth, chunk_depth))

    frameNum+=1

keypoints['Depth'] = depth
json = keypoints.to_json(orient='index')
