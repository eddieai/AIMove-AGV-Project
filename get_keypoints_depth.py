import cv2
import glob
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from initialize_OP import *
# Starting OpenPose
params['number_people_max'] = 1
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Gesture folder loacation
folderLocation = '/home/paperspace/Desktop/recording_png/hello/'
gesture = folderLocation[folderLocation[:-1].rfind('/')+1:-1]
# Sub folder number starts at
foldNum_start = 1

foldNum = foldNum_start
frameNum = 1
keypoints = []

plt.ion()
plt.figure(figsize=(20, 30))

while True:
    # Read frames on directory
    imgPath = sorted(glob.glob(folderLocation + str(foldNum) + '/*.png'),
                     key=lambda x: int(re.match(r'.*?(\d{1,3})\.png$', x).group(1)))
    imgPath_depth = sorted(glob.glob(folderLocation + str(foldNum) + 'D/*.png'),
                           key=lambda x: int(re.match(r'.*?(\d{1,3})\.png$', x).group(1)))

    img = cv2.imread(imgPath[frameNum-1])
    img_depth = cv2.imread(imgPath_depth[frameNum-1])
    print('Frame path: \t\t', imgPath[frameNum-1], '\t\t', imgPath_depth[frameNum - 1])

    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(img[..., ::-1])
    plt.subplot(2,1,2)
    plt.imshow(img_depth[..., ::-1])

    img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
    img_depth = img_depth / 255

    # Process and display images
    X = np.empty((0))
    Y = np.empty((0))
    Depth = np.empty((0))
    Probability = np.empty((0))

    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    opData = datum.poseKeypoints
    print('OpenPose keypoints shape of Folder %d Frame %d: ' % (foldNum, frameNum), datum.poseKeypoints.shape, '\n')

    for person in range(opData.shape[0]):
        for joint in range(opData.shape[1]):
            x = opData[person][joint][0].item()
            y = opData[person][joint][1].item()
            probability = opData[person][joint][2].item()
            depth = img_depth[min(img.shape[0]-1, round(y)), min(img.shape[1]-1, round(x))]

            X = np.append(X, min(img_depth.shape[1] - 1, round(x)))
            Y = np.append(Y, min(img_depth.shape[0] - 1, round(y)))
            Probability = np.append(Probability, probability)
            Depth = np.append(Depth, depth)

            keypoints.append({
                'Sub folder No.': foldNum,
                'Frame No.': frameNum,
                'Person': person,
                'Joint': joint,
                'X': x,
                'Y': y,
                'Probability': probability,
                'Depth': depth,
                })

    if not args[0].no_display:
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(15)
        if key == 27: break

    plt.subplot(2, 1, 1)
    plt.scatter(X[1:8], Y[1:8], c='white')
    for i in range(1,8):
        plt.text(X[i], Y[i], "%.02f" % Probability[i], fontdict=dict(color='white', size='15'), bbox=dict(fill=False, edgecolor='red', linewidth=1))

    plt.subplot(2, 1, 2)
    plt.scatter(X[1:8], Y[1:8], c='white')
    for i in range(1,8):
        plt.text(X[i], Y[i], "%.02f" % Depth[i], fontdict=dict(color='black', size='15'), bbox=dict(fill=False, edgecolor='red', linewidth=1))

    plt.waitforbuttonpress(0.01)
    plt.show()

    frameNum += 1
    if (frameNum>=len(imgPath)):
        foldNum += 1
        frameNum = 1
        if not (os.path.isdir(folderLocation + str(foldNum)) and os.path.isdir(folderLocation + str(foldNum) + 'D')):
            print('Sub folder %d or %s not found' % (foldNum, str(foldNum)+'D'))
            break

with open('Keypoints.Gesture_%s.SubFolder_%d-%d.json' % (gesture, foldNum_start, foldNum-1), 'w') as json_file:
    json.dump(keypoints, json_file)