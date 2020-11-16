# import the necessary packages
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# import pretrained SSD Model - set arguments
prototxt="mobilenet_ssd/MobileNetSSD_deploy.prototxt" #path to Caffe 'deploy' prototxt file
model="mobilenet_ssd/MobileNetSSD_deploy.caffemodel" #path to Caffe pre-trained model
input="videos/test.mp4" #path to optional input video file
output="output/output_01.avi" #path to optional output video file
model_confidence=0.4 #minimum probability to filter weak detections
skip_frames=30 #number of skip frames between detections


# list of class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load model from disk
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# if an input file is not specified, open the video stream
if not input:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(input)

# initialize the video writer, initialize the frame dimensions
writer = None
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers, followed by a dictionary to map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame
	frame = vs.read()
	frame = frame[1] if input else frame

	# end of the video
	if input is not None and frame is None:
		break

	# resize the frame and convert the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=350)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# set dimension of frame
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize the writer
	if output is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding box rectangles
	status = "Waiting"
	rects = []

	# check to see if we should run a detection
	if totalFrames % skip_frames == 0:
		#initialize new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > model_confidence:
				# extract the class
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")


				algorithm_tracker = cv2.TrackerCSRT_create()
				if (startY<0):
					startY=0
				if (startX<0):
					startX=0
				if (endX<0):
					endX=0
				if (endY<0):
					endY=0
				algorithm_tracker.init(frame, (startX, startY, endX, endY))

				# add the tracker to list of trackers
				trackers.append(algorithm_tracker)

	# otherwise, run tracking algorithm
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status
			status = "Tracking"

			# update the tracker and grab the updated position
			(success, box) = algorithm_tracker.update(frame)

			# unpack the position object
			(startX, startY, endX, endY) = box


			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# use the centroid tracker
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			# the difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative and the centroid is above the centerline, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# if the direction is positive and the centroid is below the center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# store the trackable object in dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
print("Down: ", totalDown)
print("Up: ", totalUp)

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not input:
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()