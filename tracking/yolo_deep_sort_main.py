import gc
from imutils.video import VideoStream
import imutils
import cv2
import sys
import os
import argparse
import numpy as np
import random
import colorsys
import time

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from yolov3_deepsort.deep_sort import preprocessing
from yolov3_deepsort.deep_sort import nn_matching
from yolov3_deepsort.deep_sort.detection import Detection
from yolov3_deepsort.deep_sort.tracker import Tracker
from yolov3_deepsort.tools import generate_detections as gdet
#import imutils.video
import tensorflow as tf

warnings.filterwarnings('ignore')

class parser(argparse.ArgumentParser):
	def __init__(self,description):
		super(parser, self).__init__(description)
		self.add_argument(
            "--outputDir", "-o",default='outputDir', type=str,
            help="path to the output directory of list files",
            metavar="<O>",
        )
		self.add_argument(
			"--video", "-v", type=str, required=False,
			help="path to input Video",
			metavar="<V>",
		)
		self.add_argument(
			"--config", "-c", type=str, required=True,
			help="path to yolo config file",
			metavar="<C>",
		)
		self.add_argument(
			"--weights", "-w", type=str, required=True,
			help="path to yolo pre-trained weights",
			metavar="<W>",
		)
		self.add_argument(
			"--classes", "-cl", type=str, required=True,
			help="path to text file containing class names",
			metavar="<CL>",
		)
		self.add_argument(
			"--confidence", "-cf", type=float, default=0.5, required=False,
			help="minimum probability to filter weak detections [default: %(default)f]",
			metavar="<CF>",
        )
		self.add_argument(
			"--use_gpu", "-u", type=bool, default=0, required=False,
			help="boolean indicating if CUDA GPU should be used",
			metavar="<U>",
        )
# ----------------------------------
# function: draw_prediction()
# ----------------------------------
def draw_prediction(img, classname, confidence, x, y, x_plus_w, y_plus_h, color,position=0):
    label =  "{}-{:.2f}%".format(classname,confidence * 100)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    if position:
        cv2.putText(img, label, (x,y_plus_h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			     
# ----------------------------------
# function: get_output_layers()
# ----------------------------------
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
# ----------------------------------
# function: detect_image_by_net()
# ----------------------------------
def detect_image_by_net(net, image, conf_threshold=0.5):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1.0/255.0
	
	# create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (inpWidth, inpHeight), (0,0,0), swapRB=True, crop=False)
	
	# set input blob for the network
    net.setInput(blob)
	
	# and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	
    return indices,class_ids,boxes,confidences

# ----------------------------------
# function: detect_tracking()
# ----------------------------------
def detect_tracking(net, image, Show_Detect=0):
    global classes,conf_threshold

    #start_time = time.time()
    
    indices,class_ids,boxes,confidences = detect_image_by_net(net,image,conf_threshold=conf_threshold)
    
    # the new ones
    new_class_ids = []
    new_confidences = []
    new_boxes = []
    for i in indices:
    	i = i[0]
    	box = boxes[i]
    	x = box[0]
    	y = box[1]
    	w = box[2]
    	h = box[3]
    	class_id=class_ids[i]
    	if Show_Detect:
    	    draw_prediction(image, classes[class_id], confidences[i], round(x), round(y), round(x+w), round(y+h),COLORS[class_id],position=1)
		    	
    	# update to new
    	new_class_ids.append(class_ids[i])
    	new_confidences.append(confidences[i])
    	new_boxes.append(boxes[i])

    names = []
    for i in range(len(new_class_ids)):
    	names.append(classes[int(new_class_ids[i])])
    names = np.array(names)
    if tracking:
    	features = encoder(image, new_boxes)
    	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(new_boxes, new_confidences, names, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    _classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxes, _classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    if tracking:
    	# Call the tracker
    	tracker.predict()
    	tracker.update(detections)

    	for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = COLORS[int(track.track_id)%classesNum]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(image, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
		
    return image

# ----------------------------------
# function: detect_camera()
# ----------------------------------
def detect_camera(net):
    vs = VideoStream(src=0).start()
    
    out = None
    if writeVideo_flag:
        fileTag = 'by_camera'
        output_video_path = os.path.sep.join([output_path,'output_dsort_'+fileTag+'.avi'])
        frame = vs.read()
        h, w = frame.shape[0], frame.shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
          
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    
    while True:
        frame = vs.read()
        #rame = imutils.resize(frame, width=400)
        t1 = time.time()
        image = detect_tracking(net, frame)
        result = np.asarray(image)  
        fps_imutils.update()
        fps = (fps + (1./(time.time()-t1))) / 2
        fps_txt = "FPS: %f" %  (fps)
        #----Text&result
        cv2.putText(result, text=fps_txt, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 255, 0), thickness=2)
        cv2.namedWindow("result")
        cv2.imshow("result", result)
        if out:
            out.write(image)
        #----
        if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
            break
        if cv2.getWindowProperty('result', 1) < 0:  # close windows using close "X" button
            gc.collect()
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
    if out:
        out.release()
    vs.stop()
    gc.collect()
    cv2.destroyAllWindows()
				
# ----------------------------------
# function: detect_video()
# ----------------------------------
def detect_video( net, video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,4*1000)
    
    out = None
    if writeVideo_flag:
        filetemp, fileExtension = os.path.splitext(video_path)
        fileTag = os.path.basename(filetemp)
        output_video_path = os.path.sep.join([output_path,'output_dsort_'+fileTag+'.avi'])
        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
	
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret != True) :
                break

        t1 = time.time()
		
        image = detect_tracking(net, frame)
        result = np.asarray(image)
        fps_imutils.update()
        fps = (fps + (1./(time.time()-t1))) / 2
        fps_txt = "FPS: %f" %  (fps)
        #----Text&result
        cv2.putText(result, text=fps_txt, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 255, 0), thickness=2)
        cv2.namedWindow("result")
        cv2.imshow("result", result)
        if out:
            out.write(image)
        #----
        if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
            break
        if cv2.getWindowProperty("result",1) < 0:  # close windows using close "X" button
            break
				
    cap.release()
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
    if out:
        out.release()
    
    cv2.destroyAllWindows()

def isFileExist(filepath,tag=""):
	result=True
	if not os.path.exists(filepath):
		print("[Error] {}:\"{}\" not found !!!".format(tag,filepath))
		result=False
	return result
def files_exist_check(files_list):
	Flag_Exit=False
	for file in files_list:
		Flag_Exit=not isFileExist(file[0],tag=file[1])
		if Flag_Exit==True:
			sys.exit()
def readTxt_byLine(filepath):
	txt = None
	with open(filepath, 'r') as f:
		txt = [line.strip() for line in f.readlines()]
	return txt
#############
#  main()
#############
args = parser(description="Detection and Tracking with OpenCV.dnn").parse_args()

files_list=[]
files_list.append([args.classes,"classes"])
files_list.append([args.config,"config"])
files_list.append([args.weights,"weights"])
if args.video:
	files_list.append([args.video,"video"])
files_exist_check(files_list)

# show information on the process ID
myProcessID=os.getpid()
idName='{}_{}'.format("PID", myProcessID)

# initialize the output/
outputDir=args.outputDir
if not os.path.exists(outputDir):
	os.makedirs(outputDir)
output_top_dir = outputDir
output_path = os.path.sep.join([output_top_dir,idName])
if not os.path.exists(output_path):
   os.makedirs(output_path)

# Model setup
RESIZED_WIDTH=416
RESIZED_HEIGHT=416
inpWidth = RESIZED_WIDTH
inpHeight = RESIZED_HEIGHT
nms_threshold = 0.4
conf_threshold = args.confidence

# read class names from text file
classes = readTxt_byLine(args.classes)

# generate different colors for different classes 
COLORS = []
classesNum=len(classes)
for x in range(classesNum):
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    rgb = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    COLORS.append(rgb)

# one dnn model
config=args.config
weights=args.weights

# read pre-trained model and config file
net = cv2.dnn.readNetFromDarknet(config, weights)

# enable CUDA
if args.use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# ----------------------------------
# tracking with deep_sort
# ----------------------------------
# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0
    
# Deep SORT
model_filename = 'yolov3_deepsort/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("[Info]GPU: set_memory_growth True")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
tracking = True
writeVideo_flag = True
	
isCamera = True 
if args.video:
    isCamera = False
    video_url = args.video
    detect_video(net, video_url)

if  isCamera :
    detect_camera(net)
    
print("[INFO] outputDir: "+output_path)