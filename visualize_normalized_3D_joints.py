import itertools as it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')


gesture, folder = 6, 9

Keypoints_1_7_normalized = pd.read_json('Keypoints_1_8_normalized_New.json', orient='records')
frame_max = Keypoints_1_7_normalized.pivot_table(index=['Gesture', 'Sub folder No.'], values=['Frame No.'], aggfunc=max).loc[gesture, folder][0]


def display_3D_keypoints(gesture, folder, frame):
    plt.cla()

    joints = Keypoints_1_7_normalized[np.logical_and.reduce((Keypoints_1_7_normalized['Gesture']==gesture, Keypoints_1_7_normalized['Sub folder No.']==folder, Keypoints_1_7_normalized['Frame No.']==frame))]
    order = np.array((4,3,2,1,5,6,7))-1

    ax.plot(joints.iloc[order]['X'], joints.iloc[order]['Y'], joints.iloc[order]['Depth'], c='w', linestyle='-', linewidth=2, alpha=0.5)
    scatter = ax.scatter(joints.iloc[0:7]['X'], joints.iloc[0:7]['Y'], joints.iloc[0:7]['Depth'], c=range(1,8), cmap='Set1', marker='o', s=40, alpha=1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    legend1 = ax.legend(*scatter.legend_elements(), title="Joint")
    ax.add_artist(legend1)

    plt.waitforbuttonpress(0.5)
    plt.show()


for frame in range(1, frame_max+1):
    display_3D_keypoints(gesture, folder, frame)