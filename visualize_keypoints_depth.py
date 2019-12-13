import cv2
import glob
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Gesture folder loacation
folderLocation = 'D:/Documents/AIMove/Project/Dataset/ASTI_labo/recording_png/hello/'
# Sub folder number starts at
foldNum_start = 1
foldNum = foldNum_start
frameNum = 1
# keypoints = []

with open('Keypoints.Gesture_%s.SubFolder_1-16.json' % (folderLocation[folderLocation[:-1].rfind('/')+1:-1]), 'r') as json_file:
    print(json_file, '\n\n')
    json_data = json.load(json_file)
json_data = iter(json_data)

plt.ion()
plt.figure(figsize=(20, 30))

while True:
    # Read frames on directory
    imgPath = sorted(glob.glob(folderLocation + str(foldNum) + '/*.png'),
                     key=lambda x: int(re.match(r'.*?(\d{1,3})\.png$', x).group(1)))
    imgPath_depth = sorted(glob.glob(folderLocation + str(foldNum) + 'D/*.png'),
                           key=lambda x: int(re.match(r'.*?(\d{1,3})\.png$', x).group(1)))

    img = cv2.imread(imgPath[frameNum-1])
    print((imgPath[frameNum-1]))
    img_depth = cv2.imread(imgPath_depth[frameNum-1])
    print((imgPath_depth[frameNum - 1]))

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

    for joint in range(25):
        data = next(json_data)
        if joint == 0:
            print('Sub folder No. ' + str(data['Sub folder No.']) + '\tFrame No. ' + str(data['Frame No.']) + '\n')
        if data['Sub folder No.'] != foldNum or data['Frame No.'] != frameNum or data['Joint'] != joint:
            raise Exception('current foldNum or frameNum or joint does not match with JSON data')

        x = data['X']
        y = data['Y']
        # depth = img_depth[min(img_depth.shape[0] - 1, round(y)), min(img_depth.shape[1] - 1, round(x))]
        depth = data['Depth']
        X = np.append(X, min(img_depth.shape[1] - 1, round(x)))
        Y = np.append(Y, min(img_depth.shape[0] - 1, round(y)))
        Depth = np.append(Depth, depth)
        Probability = np.append(Probability, data['Probability'])

        # keypoints.append({
        #     'Sub folder No.': foldNum,
        #     'Frame No.': frameNum,
        #     'Person': data['Person'],
        #     'Joint': joint,
        #     'X': x,
        #     'Y': y,
        #     'Probability': data['Probability'],
        #     'Depth': depth,
        # })

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


# with open('Keypoints.Gesture_%s.SubFolder_%d-%d.json' % (folderLocation[folderLocation[:-1].rfind('/')+1:-1], foldNum_start, foldNum-1), 'w') as json_file:
#     json.dump(keypoints, json_file)