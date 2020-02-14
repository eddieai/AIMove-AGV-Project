import pandas as pd
import glob
import os
import numpy as np

dir = '/home/aimove/Desktop/AIMove AGV Project/dataset/NEW_2_restructred/'

recording_duration = pd.read_excel('AGV_Resampling.xlsx', header=0, index_col=0)
print(recording_duration.head())

for gesture in range(1, 9):
    dir_gesture = glob.glob(dir + str(gesture) + '*')[0] + '/'

    for recording in range(1, 19):
        dir_recording_rgb = dir_gesture + str(recording) + '/'
        dir_recording_depth = dir_gesture + str(recording) + 'D/'

        frame_list = sorted(os.listdir(dir_recording_rgb), key=lambda f: int(''.join(filter(str.isdigit, f))))
        frame_resample_n = int(np.round(recording_duration.loc[recording, gesture] * 15))
        stride = int(np.round(len(frame_list) / frame_resample_n))
        frame_resample_list = frame_list[::stride]

        for frame in frame_list:
            if frame not in frame_resample_list:
                os.remove(dir_recording_rgb + frame)
                os.remove(dir_recording_depth + frame)