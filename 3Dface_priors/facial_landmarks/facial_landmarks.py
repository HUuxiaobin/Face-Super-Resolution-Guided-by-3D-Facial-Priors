# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
##write text
def write_text(five_points,name):
	with open(name, "w") as f:
		f.write(str(five_points[0][0]))
		f.write(' ')
		f.write(str(five_points[0][1]))
		f.write('\n')
		f.write(str(five_points[1][0]))
		f.write(' ')
		f.write(str(five_points[1][1]))
		f.write('\n')
		f.write(str(five_points[2][0]))
		f.write(' ')
		f.write(str(five_points[2][1]))
		f.write('\n')
		f.write(str(five_points[3][0]))
		f.write(' ')
		f.write(str(five_points[3][1]))
		f.write('\n')
		f.write(str(five_points[4][0]))
		f.write(' ')
		f.write(str(five_points[4][1]))
		f.close()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
file_path='./images'
dir=os.listdir(file_path)
dir.sort()
#print(dir)
delete_file=list()
for dir_name in dir:
	image = cv2.imread('./images/'+dir_name)
	name=dir_name.split('.')[0]
	print(name)
	# print(image.shape[1],'before')
	scale=image.shape[1]/500
	# print('this is scale',scale)
	image = imutils.resize(image, width=500)
	# print(image.shape)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	num_detect=len(rects)
	#print('rects',len(rects))
	if num_detect==0:
		print('**********please delete this file',dir_name)
		os.remove('./images/'+dir_name)
		delete_file.append(dir_name)
		#print('delete_file', delete_file)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		if i >0:
			break
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		left_eye=scale*(shape[37]+shape[41]+shape[38]+shape[40])/4
		right_eye=scale*(shape[43]+shape[44]+shape[47]+shape[46])/4
		nose_tip=scale*shape[30]
		left_mouth=scale*shape[48]
		right_mouth=scale*shape[54]
		j=0
		five_pont=[left_eye,right_eye,nose_tip,left_mouth,right_mouth]
		#print('five pints',five_pont)
		image = cv2.imread('./images/' + dir_name)
		for j in range(len(five_pont)):
			#print(five_pont[j])
			#print(int(five_pont[j][0]),int(five_pont[j][1]))
			cv2.circle(image, (int(five_pont[j][0]),int(five_pont[j][1])), 1, (0, 0, 255), -1)
			cv2.putText(image, str(j), (int(five_pont[j][0]),int(five_pont[j][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=1)
#			cv2.imshow('out',image)
#			cv2.waitKey(30000)
		# for (x, y) in shape:
		# 	j=j+1
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		# 	#cv2.putText(image, str(j), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), thickness=1)
		# 	#cv2.putText(image, str(j), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), thickness=1)

			###record num left eye and right eye center, nose tip, left and right monuth
		##caculate five key-points
		write_text(five_pont,'./land_txt/'+name+'.txt')
		#cv2.imwrite('./result/' + name + '.png', image)

# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)

# cv2.waitKey(30000)