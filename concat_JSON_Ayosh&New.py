import pandas as pd
import warnings
warnings.filterwarnings('ignore')
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

json_New = 'Keypoints_All_New.json'
json_Ayosh = 'Keypoints_All_Ayosh.json'
df_New = pd.read_json(json_New, orient='records')
df_Ayosh = pd.read_json(json_Ayosh, orient='records')
df_concat = pd.DataFrame()

df_New.loc[df_New.Gesture <= 5, 'Sub folder No.'] = df_New.loc[df_New.Gesture <= 5, 'Sub folder No.'].replace(to_replace = df_New['Sub folder No.'].unique(), value = df_New['Sub folder No.'].unique() + 16)

df_concat = pd.concat((df_New, df_Ayosh), axis=0)
new_columns = ['Gesture', 'Sub folder No.', 'Frame No.', 'Joint', 'X', 'Y', 'Probability', 'Depth']
df_concat = df_concat.reindex(columns = new_columns)
df_concat.sort_values(by=['Gesture', 'Sub folder No.', 'Frame No.', 'Joint'], inplace=True)
df_concat.reset_index(drop=True, inplace=True)

df_concat.to_json('Keypoints_All_Ayosh&New.json', orient='records')