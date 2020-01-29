import cv2
import glob
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import pandas as pd
from time import time
import warnings
warnings.filterwarnings("ignore")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
max_image_X = 640
max_image_Y = 480

config.enable_stream(rs.stream.depth, max_image_X, max_image_Y, rs.format.z16, 15)
config.enable_stream(rs.stream.color, max_image_X, max_image_Y, rs.format.bgr8, 15)
print('config :', config)

# Start streaming
pipeline.start(config)

from initialize_OP import *
# Starting OpenPose
params['number_people_max'] = 1
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Initialize list keypoints
keypoints = []

# Number of time lenght (defined pas fps)
max_frame_iter = 100

try:
    # Initialize variable iteration of frames
    frame_iter = 0

    while True:

        start_time = time()

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get frame number
        n_frame = frames.get_frame_number()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth_image to normalized depth information
        depthImg = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        img_depth = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        img_depth = img_depth / 255

        #print('frame_number :', n_frame)
        #print('color_image :', color_image)
        #print('Shape Color Image', color_image.shape)
        #print('depth_image :', img_depth)
        #print('Shape Depth Image :', img_depth.shape)

        # with open('essai_numpy.txt', 'a') as f:
        #     f.write('n_frame: ')
        #     f.write(str(n_frame))
        #     #            f.write(str(color_image.shape))
        #     #            f.write(str(depth_image.shape))
        #     f.write('max_depth_image : ')
        #     f.write((str(np.max(img_depth))))
        #     f.write('max_color_image : ')
        #     f.write((str(np.max(color_image))))
        #     f.write('  ')


        # Process and display images
        X = np.empty((0))
        Y = np.empty((0))
        Depth = np.empty((0))
        Probability = np.empty((0))

        datum = op.Datum()
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop([datum])
        opData = datum.poseKeypoints
        #print('datum.poseKeypoints.shape :', datum.poseKeypoints.shape, '\n')

        for person in range(opData.shape[0]):
            #for joint in range(opData.shape[1]):
            # Only joint 1 to 7
            for joint in range(1, 8):
                x = opData[person][joint][0].item() / max_image_X
                y = opData[person][joint][1].item() / max_image_Y
                probability = opData[person][joint][2].item()
                depth = img_depth[min(color_image.shape[0]-1, round(y)), min(color_image.shape[1]-1, round(x))]

                X = np.append(X, min(img_depth.shape[1] - 1, round(x)))
                Y = np.append(Y, min(img_depth.shape[0] - 1, round(y)))
                Probability = np.append(Probability, probability)
                Depth = np.append(Depth, depth)

                keypoints.append({
                    'Frame No.': n_frame,
                    'Person': person,
                    'Joint': joint,
                    'X': x,
                    'Y': y,
                    'Probability': probability,
                    'Depth': depth,
                    })

        #Vizualization of Keypoints
        #print('keypoints :', n_frame, keypoints)


        if not args[0].no_display:
             cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
             key = cv2.waitKey(15)
             if key == 27: break

        frame_iter = frame_iter + 1
        print('frame_iter :', frame_iter)
        print('n_frame :', n_frame)

        if frame_iter == max_frame_iter:
            # Create DataFrame of keypoints
            df=pd.DataFrame.from_dict(keypoints, orient='columns')
            print('df :',df)

            end_time = time()
            run_time = end_time - start_time
            print(run_time)

            # Update variable iteration of frames
            frame_iter = max_frame_iter - 1
            # Delete first keypoints (first frame of the dataframe)
            del keypoints[:7]

finally:
    # Stop streaming
     pipeline.stop()