import cv2
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Gesture folder loacation
folderLocation = 'D:/Documents/AIMove/Project/Dataset/ASTI_labo/recording_png/hello/'
# Sub folder number starts at
foldNum_start = 1
foldNum = foldNum_start
frameNum = 1
keypoints = []

with open('Keypoints.Gesture_%s.SubFolder_1-16.json' % (folderLocation[folderLocation[:-1].rfind('/')+1:-1]), 'r') as json_file:
    json_data = json.load(json_file)
json_data = iter(json_data)

plt.ion()
plt.figure()

while True:
    # Read frames on directory
    imgPath = glob.glob(folderLocation + str(foldNum) + '/*.png')
    imgPath_depth = glob.glob(folderLocation + str(foldNum) + 'D/*.png')
    if (frameNum>=len(imgPath_depth)):
        foldNum += 1
        frameNum = 1
        if not os.path.isdir(folderLocation + str(foldNum) + 'D'):
            print('Sub folder %s not found' % (str(foldNum)+'D'))
            break
        imgPath_depth = glob.glob(folderLocation + str(foldNum) + 'D/*.png')

    img = cv2.imread(imgPath[frameNum-1])
    img_depth = cv2.imread(imgPath_depth[frameNum-1])

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
        print('foldNum: ' + str(data['Sub folder No.']) + '\tframeNum: ' + str(data['Frame No.']) + '\tJoint: ' + str(data['Joint']))
        if data['Sub folder No.'] != foldNum or data['Frame No.'] != frameNum or data['Joint'] != joint:
            raise Exception('current foldNum or frameNum or joint does not match with JSON data')

        x = data['X']
        y = data['Y']
        # depth = np.empty((9))
        # depth_ind = 0
        # for y_shift in [-1,0,1]:
        #     for x_shift in [-1,0,1]:
        #         depth[depth_ind] = img_depth[min(img_depth.shape[0]-1, round(y+y_shift)), min(img_depth.shape[1]-1, round(x+x_shift))]
        #         depth_ind += 1
        depth = img_depth[min(img_depth.shape[0] - 1, round(y)), min(img_depth.shape[1] - 1, round(x))]
        X = np.append(X, min(img_depth.shape[1] - 1, round(x)))
        Y = np.append(Y, min(img_depth.shape[0] - 1, round(y)))
        Depth = np.append(Depth, depth)
        Probability = np.append(Probability, data['Probability'])

        keypoints.append({
            'Sub folder No.': foldNum,
            'Frame No.': frameNum,
            'Person': data['Person'],
            'Joint': joint,
            'X': x,
            'Y': y,
            'Probability': data['Probability'],
            'Depth': depth,
            })

    plt.subplot(2, 1, 1)
    plt.scatter(X[1:8], Y[1:8], c='white')
    for i in range(1,8):
        plt.text(X[i], Y[i], "%.02f" % Probability[i], fontdict=dict(color='white', size='10'), bbox=dict(fill=False, edgecolor='red', linewidth=1))

    plt.subplot(2, 1, 2)
    plt.scatter(X[1:8], Y[1:8], c='white')
    for i in range(1,8):
        plt.text(X[i], Y[i], "%.02f" % Depth[i], fontdict=dict(color='white', size='10'), bbox=dict(fill=False, edgecolor='red', linewidth=1))

    plt.waitforbuttonpress()
    plt.show()

    frameNum += 1


# with open('Keypoints_Depth.Gesture_%s.SubFolder_%d-%d.json' % (folderLocation[folderLocation[:-1].rfind('/')+1:-1], foldNum_start, foldNum-1), 'w') as json_file:
#     json.dump(keypoints, json_file)