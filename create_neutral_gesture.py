import pandas as pd
import numpy as np
import glob
import shutil
import os

dir = '/home/aimove/Desktop/AIMove AGV Project/dataset/NEW_2_restructred/'

neutral_frames = pd.read_excel('AGV_DATASET-NEUTRAL.xlsx', header=0, index_col=0)
dir_neutral = glob.glob(dir + '8*')[0] + '/'


def create_neutral_gesture():
    for gesture in range(1,8):
        dir_gesture = glob.glob(dir + str(gesture) +'*')[0] + '/'

        for recording in range(1,19):
            dir_recording_rgb = dir_gesture + str(recording) + '/'
            dir_recording_depth = dir_gesture + str(recording) + 'D/'

            list_frame = sorted(os.listdir(dir_recording_rgb), key=lambda f:  int(''.join(filter(str.isdigit, f))))

            non_neutral_from = neutral_frames.loc[recording, str(gesture)+'_from']
            non_neutral_to = neutral_frames.loc[recording, str(gesture)+'_to']

            for frame in list_frame[:non_neutral_from] + list_frame[non_neutral_to:]:
                if not os.path.exists(dir_neutral + str(recording) + '/' + str(gesture)):
                    os.makedirs(dir_neutral + str(recording) + '/' + str(gesture))
                shutil.move(dir_recording_rgb + frame, dir_neutral + str(recording) + '/' + str(gesture))

                if not os.path.exists(dir_neutral + str(recording) + 'D/' + str(gesture)):
                    os.makedirs(dir_neutral + str(recording) + 'D/' + str(gesture))
                shutil.move(dir_recording_depth + frame, dir_neutral + str(recording) + 'D/' + str(gesture))


def merge_neutral_recording():
    for recording in range(1,19):
        dir_recording_rgb = dir_neutral + str(recording) + '/'
        dir_recording_depth = dir_neutral + str(recording) + 'D/'

        frame_count = 1
        for gesture in range(1,8):
            dir_gesture_rgb = dir_recording_rgb + str(gesture) + '/'
            dir_gesture_depth = dir_recording_depth + str(gesture) + '/'

            list_frame = sorted(os.listdir(dir_gesture_rgb), key=lambda f:  int(''.join(filter(str.isdigit, f))))

            for frame in list_frame:
                shutil.move(dir_gesture_rgb + str(frame), dir_recording_rgb + str(frame_count) + '.png')
                shutil.move(dir_gesture_depth + str(frame), dir_recording_depth + str(frame_count) + '.png')
                frame_count += 1