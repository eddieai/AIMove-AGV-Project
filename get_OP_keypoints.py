import cv2
import glob
import os
import json
from initialize_OP import *

# Starting OpenPose
params['number_people_max'] = 1
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Gesture folder loacation
folderLocation = '/home/paperspace/Desktop/recording_png/speed_up/'
# Sub folder number starts at
foldNum_start = 1
foldNum = foldNum_start
frameNum = 1
keypoints = []


while True:
    # Read frames on directory
    imgPath = glob.glob(folderLocation+str(foldNum)+'/*.png')
    imgPath_depth = glob.glob(folderLocation + str(foldNum) + 'D/*.png')
    if (frameNum>=len(imgPath)):
        foldNum += 1
        frameNum = 1
        if not (os.path.isdir(folderLocation + str(foldNum)) and os.path.isdir(folderLocation + str(foldNum) + 'D')):
            print('Sub folder %d or %s not found' % (foldNum, str(foldNum)+'D'))
            break
        imgPath = glob.glob(folderLocation+str(foldNum)+'/*.png')
        imgPath_depth = glob.glob(folderLocation + str(foldNum) + 'D/*.png')

    img = cv2.imread(imgPath[frameNum-1])
    img_depth = cv2.imread(imgPath_depth[frameNum-1])
    img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
    img_depth = img_depth / 255

    # Process and display images
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    opData = datum.poseKeypoints
    print('Keypoints shape of Folder %d Frame %d: ' % (foldNum, frameNum), datum.poseKeypoints.shape)

    for person in range(opData.shape[0]):
        for joint in range(opData.shape[1]):
            x = opData[person][joint][0].item()
            y = opData[person][joint][1].item()
            probability = opData[person][joint][2].item()
            depth = img_depth[min(479, round(y)), min(639, round(x))]

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

    frameNum += 1


with open('Keypoints.Gesture_%s.SubFolder_%d-%d.json' % (folderLocation[folderLocation[:-1].rfind('/')+1:-1], foldNum_start, foldNum-1), 'w') as json_file:
    json.dump(keypoints, json_file)