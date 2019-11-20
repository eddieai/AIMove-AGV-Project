# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import pyrealsense2 as rs
import argparse
import time
import numpy as np
import glob
import argparse
import random
import time

# from pythonosc import osc_message_builder
# from pythonosc import udp_client
slidingWindow_length = 30
usedJoints = [4,7]
UDP_IP = "127.0.0.1"
UDP_PORT = 7374
MESSAGE = "Hello, World!"
UDP=1
resX=640
resY=480
#33
folderLocation='/home/aimove/Desktop/Glass_Blowing_New/Blow_through_the_stick/'

posture_Final=[]

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                        help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=7374,
                        help="The port the OSC server is listening on")
args = parser.parse_args()

# client = udp_client.SimpleUDPClient(args.ip, args.port)




# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
parser.add_argument("--camera_resolution 640x480")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments

# op.init_argv(args[1])
# oppython = op.OpenposePython()

    # Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
'''
# REALSENSE Configure depth and color streams
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


'''

# Realtime recognition
slidingWindow = []
foldNum=1
frameNum=1
while True:
    '''
    start_time = time.time() # start time of the loop
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    '''

    
        
    imgPath=glob.glob(folderLocation+str(foldNum)+'/*.png')
    if (frameNum>len(imgPath)):
        frameNum=0
        foldNum+=1
        imgPath=glob.glob(folderLocation+str(foldNum)+'/*.png')
    img = cv2.imread(imgPath[frameNum-1])
    frameNum+=1
    #print(imgPath)
    #imgPath=folderLocation+str(foldNum)+'/frame'+str(frameNum)+'.jpg'
    #print(os.path.isfile(imgPath))
    '''                 
    if os.path.isfile(imgPath)==True:
        img = cv2.imread(imgPath[frameNum-1])
        if(foldNum<=33):
            img=img[:, int(2704/2)-200:int(2704/2)+350].copy()
        else:
            img=img[:, int(2704/2)-600:2703].copy()
        frameNum+=1
    else:
        sys.stdout=open("txtFiles3/gesture3_"+str(foldNum)+".txt","w")
        for i in posture_Final:
           print(i)
        sys.stdout.close()
        foldNum+=1
        frameNum=1
        imgPath=folderLocation+str(foldNum)+'/frame'+str(frameNum)+'.jpg'
        img = cv2.imread(imgPath)
        if(foldNum<=33):
            img=img[:, int(2704/2)-200:int(2704/2)+350].copy()
        else:
            img=img[:, int(2704/2)-600:2703].copy()
        posture_Final=[]
    '''   
    #print(img)
    datum = op.Datum()
    datum.cvInputData = img
    
    opWrapper.emplaceAndPop([datum])
    outputFrame = datum.cvOutputData

    
   
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,400)
    fontScale = 2
    fontColor = (255,255,255)
    lineType = 2
    # Getting OpenPose output
    opData = datum.poseKeypoints
    print(datum.poseKeypoints.shape)
    
    if (opData.size == 1):
        continue
    #centralJointX=abs(320-opData[0][1][0])
    #centralJointY=abs(240-opData[0][1][1])
    centralJointX=opData[0][1][0]
    print(centralJointX, 'centralJointX')
    centralJointY=opData[0][1][1]
    print(centralJointY, 'centralJointY')
   
    #centralJointZ=aligned_depth_frame.get_distance(opData[0][1][0],opData[0][1][1])
    #centralJointZ=0
    '''
    if(foldNum<=33):
        resX=2703+350-int(2704/2)+200
    else:
        resX=2703-int(2704/2)+600

    '''
    posture = np.concatenate([[(opData[0][i][0]-centralJointX)/resX, (opData[0][i][1]-centralJointY)/resY] if (int(opData[0][i][0])!=643) else [(opData[0][i][0]
                                                                                       -centralJointX)/resX, (opData[0][i][1]-centralJointY)/resY] for i in usedJoints ])
    print(posture, 'posture')

    '''


    posture = np.concatenate([[opData[0][i][0], opData[0][i][1],
                               aligned_depth_frame.get_distance(opData[0][i][0],opData[0][i][1])
                               ] if (int(opData[0][i][0])!=643) else [opData[0][i][0], opData[0][i][1]-centralJointY,aligned_depth_frame.get_distance(opData[0][i][0],opData[0][i][1])] for i in usedJoints ])
    '''
    '''
    roundX=abs(centralJointX)/640
    roundY=abs(centralJointY)/480
    for i in range(len(usedJoints)*3-2):
        
        if posture[i]==roundX and posture[i+1]==roundY:
            posture[i:i+3]=0
       
            
    '''        
    #posture2 = np.concatenate([[opData[0][i][0], opData[0][i][1], depth_image.get_distance(int(opData[0][i][0]),int(opData[0][i][1]))] for i in usedJoints if (int(opData[0][i][0])!=643)])

    s=str(posture)
    ss=s.replace('[', '')
    sss=ss.replace(']', '')
    #dd=aligned_depth_frame.get_distance(opData[0][4][0],opData[0][4][1])
    #pixel_distance_in_meters = dpt_image.get_distance(x,y)
    '''
    client.send_message("/RHx", float(posture[0]))
    client.send_message("/RHy", float(posture[1]))
    client.send_message("/RHz", float(posture[2]))
    client.send_message("/LHx", float(posture[3]))
    client.send_message("/LHy", float(posture[4]))
    client.send_message("/LHz", float(posture[5]))
    '''
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)
    

    
    

    #print(posture)
    
    #cv2.circle(depth_colormap,(opData[0][1][0],opData[0][1][1]), 10, (255,255,255), 10)
    #images = np.hstack((outputFrame, depth_colormap))

        # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', outputFrame)
    posture_Final.append(sss)

    
    #print(depth_image.shape,depth_colormap.shape)
    k=cv2.waitKey(1)
    if k == 27 :
        '''# wait for ESC key to exit
        sys.stdout=open("txtFiles3/gesture3_"+str(foldNum)+".txt","w")
        for i in posture_Final:
           print(i)
        sys.stdout.close()
        '''
        cv2.destroyAllWindows()
        break
        
    #print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop



