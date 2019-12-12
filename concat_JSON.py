import glob
import pandas as pd
import numpy as np
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
np.set_printoptions(linewidth=desired_width)

jsonPath = glob.glob('./*Gesture*.json')
keypoints_all = pd.DataFrame()
for jsonfile in jsonPath:
    gesture = jsonfile[jsonfile.find('Gesture')+8:jsonfile.find('.SubFolder')]
    df = pd.read_json(jsonfile, orient='records')
    df['Gesture'] = gesture
    keypoints_all = pd.concat((keypoints_all, df), axis=0)

gesture_list = keypoints_all['Gesture'].unique()
i = 1
for gesture_name in gesture_list:
    keypoints_all = keypoints_all.replace(gesture_name, i)
    i+=1
print('Gesture Name, Gesture No. :\n', list(zip(gesture_list,range(1,len(gesture_list)+1))))
#  ('Blow_through_the_stick', 1), ('Tighten_the_base_of_the_glass_with_the_pliers', 2), ('New_Gesture4', 3)
#  ('hello', 1), ('left', 2), ('right', 3), ('speed_down', 4), ('speed_up', 5)

keypoints_all.to_json(r'.\Keypoints_All.json', orient='records')