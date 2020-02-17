import cv2
import glob
import json
import os
import re
import numpy as np
from numpy import genfromtxt
import csv
from tslearn.metrics import dtw
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import pandas as pd
from time import time
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
max_image_X = 640
max_image_Y = 480
config.enable_stream(rs.stream.depth, max_image_X, max_image_Y, rs.format.z16, 15)
config.enable_stream(rs.stream.color, max_image_X, max_image_Y, rs.format.bgr8, 15)


# Start streaming
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

from initialize_OP import *

# Starting OpenPose
params['number_people_max'] = 1
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# List of Gestures
gesture_index = ["1_START", "2_RIGHT", "3_LEFT", "4_SPEED-DOWN", "5_SPEED-UP", "6_CONFIRMATION", "7_STOP", "8_NEUTRAL"]

# Read HMM trained model parameters from JSON
with open('hmm_8020_diag_trained_param.json', 'r') as json_file:
    trained_param = json.load(json_file)

# Recover HMM models from HMM trained model parameters
hmm_model = []
for i in range(8):
    hmm_model.append(hmm.GaussianHMM(n_components=5, covariance_type='full'))
    hmm_model[-1].startprob_ = np.array(trained_param['startprob'][i])
    hmm_model[-1].transmat_ = np.array(trained_param['transmat'][i])
    hmm_model[-1].means_ = np.array(trained_param['means'][i])
    hmm_model[-1].covars_ = np.array(trained_param['covars'][i])
    hmm_model[-1].n_features = 21

# Initialize numpy matrix of result
data_window = np.zeros((0, 7, 4))

# Number of time lenght (defined pas fps)
max_frame_iter = 60
# Number of frame slider
frame_slide = 10

plt.ion()
fig = plt.figure()


def depth_cleaned(data_window, iter = 1):
    for _ in range(iter):
        depth_median = np.median(data_window[:, :, 3], axis=0)
        depth_median_upper = depth_median + 0.1
        depth_median_lower = depth_median - 0.1

        depth_frame_before = np.roll(data_window[:, :, 3], 1, axis=0)
        depth_frame_after = np.roll(data_window[:, :, 3], -1, axis=0)

        depth_to_replace = np.logical_or(data_window[:, :, 3] < depth_median_lower,
                                         data_window[:, :, 3] > depth_median_upper)

        data_window[:, :, 3] = data_window[:, :, 3] * np.invert(depth_to_replace) + depth_to_replace * (
                    (depth_frame_before + depth_frame_after) / 2)
    return data_window


def dtw_classifier(data_window_distance):

    dtw_train_csv = list(csv.reader(open('/home/aimove/Desktop/AIMove AGV Project/AIMove-AGV-Project/dtw_data.csv')))
    dtw_train_data = []
    for row in dtw_train_csv:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        dtw_train_data.append(nwrow)
    dtw_train_data = np.array(dtw_train_data)
    print(dtw_train_data.shape)
    dist = np.empty(8)
    for i in range(8):
        dist[i] = dtw(np.array(dtw_train_data[i]), data_window_distance)
    dtw_pred = np.argmin(dist)

    print(gesture_index[dtw_pred])
    return ()


def hmm_classifier(data_window_distance):

    start_time = time()

    score = np.empty(8)
    for k in range(8):
        try:
            score[k] = hmm_model[k].score(data_window_distance, lengths=[len(data_window_distance)])
        except:
            score[k] = -99999

    predict_idx = np.argmax(score)    # the same index of predict idx and gesture index
    predict = gesture_index[predict_idx]

    score_scaled = np.interp(score, (score.min(), score.max()), (0,10))
    end_time = time()

    if max(score) > -9999999:
        print('score: ', score)
        # print('log_Likelihood to prob: ', score_scaled / score_scaled.sum())
        # print('log_to_exp: ', np.exp(score_scaled))
        # print('log_to_exp to prob: ', 1*np.exp(score_scaled)/(np.exp(score_scaled).sum()))
        print(predict)

    return

try:
    # Initialize variable iteration of frames
    frame_iter = 0

    while True:

        start_time = time()

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Get frame number
        n_frame = frames.get_frame_number()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # plt.clf()
        # plt.subplot(2, 1, 1)
        # plt.imshow(color_image)
        # plt.subplot(2, 1, 2)
        # plt.imshow(depth_image)

        # Convert depth_image to normalized depth information
        depthImg = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        img_depth = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        img_depth = img_depth / 255

        # Process and display images
        X = np.empty((0))
        Y = np.empty((0))
        Depth = np.empty((0))
        Probability = np.empty((0))

        datum = op.Datum()
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop([datum])
        opData = datum.poseKeypoints
        # print('datum.poseKeypoints.shape :', datum.poseKeypoints.shape, '\n')

        X = np.empty((0))
        Y = np.empty((0))
        Depth = np.empty((0))
        Probability = np.empty((0))

        data_joint = np.empty((0, 4))
        for joint in range(1, 8):
            x = opData[0][joint][0].item()
            y = opData[0][joint][1].item()
            probability = opData[0][joint][2].item()
            depth = img_depth[min(color_image.shape[0] - 1, round(y)), min(color_image.shape[1] - 1, round(x))]

            X = np.append(X, min(img_depth.shape[1] - 1, round(x)))
            Y = np.append(Y, min(img_depth.shape[0] - 1, round(y)))
            Probability = np.append(Probability, probability)
            Depth = np.append(Depth, depth)

            data_keypoint = np.array([x / max_image_X, y / max_image_Y, probability, depth])
            data_joint = np.vstack((data_joint, data_keypoint))

        data_window = np.vstack((data_window, data_joint.reshape(1, 7, 4)))

        # Plot the current depth (visualization)
        # plt.subplot(2, 1, 1)
        # plt.scatter(X, Y, c='white')
        # for i in range(0,7):
        #     plt.text(X[i], Y[i], "%.02f" % Probability[i], fontdict=dict(color='white', size='15'), bbox=dict(fill=False, edgecolor='red', linewidth=1))
        #
        # plt.subplot(2, 1, 2)
        # plt.scatter(X, Y, c='white')
        # for i in range(0,7):
        #     plt.text(X[i], Y[i], "%.02f" % Depth[i], fontdict=dict(color='black', size='15'), bbox=dict(fill=False, edgecolor='red', linewidth=1))
        #
        # plt.waitforbuttonpress(0.0001)
        # plt.show()

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break

        frame_iter = frame_iter + 1
        # print('frame_iter :', frame_iter, '\tn_frame :', n_frame)

        if frame_iter == max_frame_iter:
            # Create DataFrame of keypoints
            # print('data window shape:', data_window.shape)
            end_time = time()
            run_time = end_time - start_time
            print('run time', run_time)

            data_window = depth_cleaned(data_window, iter=5)
            # data_window_distance = np.empty([max_frame_iter,1])
            data_window_distance = np.empty([max_frame_iter,0])


            for pair in list(combinations(np.arange(0,7), 2)):
                x1 = data_window[:, pair[0], 0]
                x2 = data_window[:, pair[1], 0]
                y1 = data_window[:, pair[0], 1]
                y2 = data_window[:, pair[1], 1]
                depth1 = data_window[:, pair[0], 3]
                depth2 = data_window[:, pair[1], 3]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (depth1 - depth2)**2)
                # data_window_distance = np.hstack((data_window_distance, distance.reshape(1,max_frame_iter).T))
                data_window_distance = np.hstack((data_window_distance, distance.reshape(max_frame_iter, 1)))

            # data_window_distance = data_window_distance[:,1:] / np.max(data_window_distance[:,1:])
            data_window_distance = data_window_distance / np.max(data_window_distance)

            # print(data_window_distance)

            # DTW
            dtw_classifier(data_window_distance)

            # HMM
            # hmm_classifier(data_window_distance)

            # Update variable iteration of frames
            frame_iter = max_frame_iter - frame_slide

            # Delete first keypoints (first frame of the dataframe)
            data_window = data_window[frame_slide:, :, :]

            # Print FPS
            print("FPS: ", 1.0 / (time() - start_time))

finally:
    # Stop streaming
    pipeline.stop()
