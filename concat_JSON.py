import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

jsonPath = sorted(glob.glob('Keypoints.Gesture_*.json'))
print('Keypoints JSON path:')
[print('\t- ', _) for _ in jsonPath]

keypoints_all = pd.DataFrame()
for jsonfile in jsonPath:
    gesture = jsonfile[jsonfile.find('Gesture')+8:jsonfile.find('.SubFolder')]
    df = pd.read_json(jsonfile, orient='records')
    df['Gesture'] = gesture
    keypoints_all = pd.concat((keypoints_all, df), axis=0)

i = 1
gesture_list = keypoints_all['Gesture'].unique()
for gesture in gesture_list:
    keypoints_all = keypoints_all.replace(gesture, i)
    i += 1
gesture_index = pd.DataFrame([[_] for _ in range(1, len(gesture_list)+1)], index=gesture_list, columns=['Gesture index'])
print('\n', gesture_index)

gesture_index.to_json('Gesture_index.json', orient='columns')
keypoints_all.to_json('Keypoints_All.json', orient='records')