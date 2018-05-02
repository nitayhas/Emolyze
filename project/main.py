###################################
#
#	Written by: Nitay Hason
#	Email: nitay.has@gmail.com
#
###################################
from statistics import mode
import dlib
import numpy as np
import face_recognition
import cv2
import glob,json,os,math
from time import time

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import decimal

from os.path import basename
from state_vec import StateVector
from mydb import Dynamodb
from keras.models import load_model

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

dynamodb = Dynamodb()
table = dynamodb.getTable('People')

# parameters for loading data and images
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)
frame_size = 0.5
duration = 100
save=0

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)
video_capture.set(4,800)

image_files = glob.glob('./students/*.jpg')

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_ids = []
known_face_names = []

# Load a picture and learn how to recognize it.
for image in image_files:
	user_image = face_recognition.load_image_file(image)
	user_face_encoding = face_recognition.face_encodings(user_image)[0]
	known_face_encodings.append(user_face_encoding)
	person_id = os.path.splitext(basename(image))[0]
	known_face_ids.append(person_id)
	item = dynamodb.getItem(table,{'PersonId': person_id})

	if(item is not None):
		known_face_names.append(item["Name"])
	else:
		known_face_names.append(person_id)

state_vector = StateVector()


while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()
	send_to_db = False
	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	# rgb_frame = frame[:, :, ::-1]
	small_frame = cv2.resize(frame, (0, 0), fx=frame_size, fy=frame_size)
	rgb_frame = frame
	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Find all the faces and face enqcodings in the frame of video
	face_locations = face_recognition.face_locations(small_frame)
	face_encodings = face_recognition.face_encodings(small_frame, face_locations)

	# Loop through each face in this frame of video
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		# If a match was found in known_face_encodings, just use the first one.
		if True in matches:
			name = "Unknown"
			face_coordinates = (left, top, right-left, bottom-top)

			frame_size_mult = int(math.pow(frame_size, -1))
			left=frame_size_mult*left
			right=frame_size_mult*right
			top=frame_size_mult*top
			bottom=frame_size_mult*bottom

			gray_face = gray_image[top:bottom, left:right]

			try:
				gray_face = cv2.resize(gray_face, (emotion_target_size))
			except:
				continue

			gray_face = preprocess_input(gray_face, True)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_prediction = emotion_classifier.predict(gray_face)
			emotion_probability = np.max(emotion_prediction)
			emotion_label_arg = np.argmax(emotion_prediction)
			emotion_text = emotion_labels[emotion_label_arg]
			emotion_window.append(emotion_text)

			if len(emotion_window) > frame_window:
				emotion_window.pop(0)
			try:
				emotion_mode = mode(emotion_window)
			except:
				continue

			if emotion_text == 'angry':
				color = emotion_probability * np.asarray((255, 0, 0))
			elif emotion_text == 'sad':
				color = emotion_probability * np.asarray((0, 0, 255))
			elif emotion_text == 'happy':
				color = emotion_probability * np.asarray((255, 255, 0))
			elif emotion_text == 'surprise':
				color = emotion_probability * np.asarray((0, 255, 255))
			else:
				color = emotion_probability * np.asarray((0, 255, 0))

			color = color.astype(int)
			color = color.tolist()

			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]
			person_id = known_face_ids[first_match_index]
			send_to_db = state_vector.add(person_id, emotion_label_arg, duration)
			if send_to_db:
				length = state_vector.length(person_id,emotion_label_arg)
				if length>0:
					cur_table = dynamodb.getTable('StatesData')
					popped = state_vector.popleft(person_id,emotion_label_arg)

					item={
				        'StateDataId': str(time()),
						'PersonId': person_id,
				        'State': str(emotion_label_arg),
				        'Duration': decimal.Decimal(popped['duration']),
						'StartTime': decimal.Decimal(popped['start_time'])
				    }
					dynamodb.putItem(cur_table, item)

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
			cv2.putText(frame, emotion_mode, (left + 6, top - 45), font, 1.0, color, 1)
			# print(state_vector.get())


	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	k = cv2.waitKey(duration)
	if k & 0xFF == ord('q'):
		break
	elif k & 0xFF == ord('p'):
		cv2.imwrite("./saves/image_processed_%d.jpg" % save, frame)
		save+=1

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
