import pyrealsense2 as rs
import cv2
import numpy as np
import os

def convert(filename, rgbPath, depthPath):
	print('Begin conversion of ' + filename)
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
	config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
	config.enable_device_from_file(filename, False)
	profile = pipeline.start(config)
	playback = profile.get_device().as_playback()
	playback.set_real_time(False)

	align_to = rs.stream.color
	align = rs.align(align_to)
	colorizer = rs.colorizer(3)

	try:
		i = 0
		while True:
			i += 1
			print(i)
			frames = pipeline.wait_for_frames()
			frames = align.process(frames)
			color = frames.get_color_frame()
			depth = frames.get_depth_frame()

			depth = colorizer.colorize(depth)

			if (not(depth) or not(color)):
				continue

			depth_color_frame = colorizer.colorize(depth)
			frame = np.asanyarray(color.get_data())
			dFrame = np.asanyarray(depth_color_frame.get_data())

			#depthImg = cv2.applyColorMap(cv2.convertScaleAbs(dFrame, alpha=0.03), cv2.COLORMAP_JET)
			rgbImg = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			cv2.imwrite(rgbPath + str(i) + ".png", rgbImg)
			cv2.imwrite(depthPath + str(i) + ".png", dFrame)
	except RuntimeError:
		print('Finish conversion of ' + filename + '\n\n')
	finally:
		pipeline.stop()

dataset_dir = 'D:/Documents/AIMove/Project/Dataset/ASTI_labo/'
png_dir = 'D:/Documents/AIMove/Project/Dataset/ASTI_labo/recording_png/'

for root, dirs, files in os.walk(dataset_dir):
	for name in files:
		if name.endswith(".bag"):
			bag_path = os.path.join(root, name)

			gesture = root.split('\\')[-2]
			png_path = os.path.join(png_dir, gesture)
			if os.path.exists(png_path):
				sub_folder = len(os.listdir(png_path))//2 + 1
			else:
				os.mkdir(png_path)
				sub_folder = 1

			rgb_dir = png_path + '/' + str(sub_folder) + '/'
			os.mkdir(rgb_dir)
			depth_dir = png_path + '/' + str(sub_folder) + 'D/'
			os.mkdir(depth_dir)

			convert(bag_path, rgb_dir, depth_dir)