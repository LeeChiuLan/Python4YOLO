
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
import imutils.video
import tensorflow as tf

warnings.filterwarnings('ignore')

RESIZED_WIDTH=416
RESIZED_HEIGHT=416

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
# class: removeDir
# ----------------------------------
class removeDir:
   def __init__(self,dir):
      for root, dirs, files in os.walk(dir, topdown=False):
         for name in files:
            os.remove(os.path.join(root, name))
         for name in dirs:
            os.rmdir(os.path.join(root, name))
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
# function: getIndicesByID()
# ----------------------------------
def getIndicesByID(indices, class_ids,classesNum,keyID):
    outStacks =[[] for x in range( classesNum )]
    #print("classesNum="+str(classesNum))
    stacks=[[] for x in range( classesNum )]
    for i in indices:
        stacks[class_ids[i[0]]].append(i)
    for k in stacks[keyID]:
        i = k[0]
        outStacks[keyID].append(k)
                
    return outStacks[keyID]
# ----------------------------------
# class: NetBlob
# ----------------------------------
class NetBlob:
    def __init__(self,classid,classname,indices,class_ids,boxes,confidences,classesNum,color=(0,0,0),callback=[]):
    	self.classid=classid
    	self.name=classname
    	self.indices=indices
    	self.class_ids=class_ids
    	self.boxes=boxes
    	self.confidences=confidences
    	self.classesNum=classesNum
    	self.color=color
    	self.getMyInices()
    	
    	self.callback=callback
    	self.Database=None

    def setcolor(self,color):
    	self.color=color
    	
    def getMyInices(self):
    	self.indices=getIndicesByID(self.indices, self.class_ids,self.classesNum,self.classid)
		
    def getInices(self):
    	num=0
    	for i in self.indices:
    		num = num + 1
    	#if num == 0:
    		#num = None
    	return self.indices, num

    def draw(self,img):
    	for i in self.indices:
    		i = i[0]
    		box = self.boxes[i]
    		x = round(box[0])
    		y = round(box[1])
    		w = round(box[2])
    		h = round(box[3])
    		confidence=self.confidences[i]
    		draw_prediction(img, self.name, confidence, x, y, x+w, y+h, self.color)
    def setCallback(self,callback):
    	self.callback=callback
    		
    def executeActions(self):
    	for func in zip(self.callback):
    		func()
# ----------------------------------
# class: BBall
# ----------------------------------
class BBall(NetBlob):
	def executeActions(self,image=None,showOn=1):
		super().executeActions()
		self.oldXY=None
		for i in self.indices:
			i = i[0]
			box = self.boxes[i]
			x = round(box[0])
			y = round(box[1])
			w = round(box[2])
			h = round(box[3])
			self.oldXY=[int(x+w/2),int(y+h/2),w,h]
			if image is not None and self.oldXY is not None:
				x=self.oldXY[0]
				y=self.oldXY[1]
				w=self.oldXY[2]
				h=self.oldXY[3]
				cv2.circle(image,((int)(x),(int)(y)), 5, (0,0,255), -1)
				label =  "({},{})".format(x,y)
				if showOn: print("({})x,y={}".format(self.__class__.__name__,label))
				cv2.putText(image, label, (x,y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
			     
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
def detect_tracking(net, image, showOn=1, writeVideo=None):
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

    #for det in detections:
    #	bbox = det.to_tlbr()
    #	score = "%.2f" % round(det.confidence * 100, 2) + "%"
    #	cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    #	if len(_classes) > 0:
    #        cls = det.class_name
    #        cv2.putText(image, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
	#					1.5e-3 * image.shape[0], (0, 255, 0), 1)

    if writeVideo:
    	writeVideo.write(image)
		
    return image
# ----------------------------------
# function: detect_image()
# ----------------------------------
def detect_image(net, image, showOn=1):
    global classes,conf_threshold

    #start_time = time.time()
    
    indices,class_ids,boxes,confidences = detect_image_by_net(net,image,conf_threshold=conf_threshold)
    indices2,class_ids2,boxes2,confidences2 = indices,class_ids,boxes,confidences
    if Court_id is not None:
    	court_netblob=NetBlob(Court_id,classes[Court_id],indices2,class_ids2,boxes2,confidences2,classesNum,COLORS[Court_id])
    	court_netblob.draw(image)
    	
    ball_netblob=BBall(Ball_id,classes[Ball_id],indices,class_ids,boxes,confidences,classesNum,COLORS[Ball_id])
    #ball_netblob.executeActions(image,showOn=showOn)
    ball_netblob.draw(image)
    
    #print("--- %s seconds ---" % (time.time() - start_time))	
    return image
# ----------------------------------
# function: detect_camera()
# ----------------------------------
def detect_camera(net):
    vs = VideoStream(src=0).start()
    fps = ""
    while True:
        frame = vs.read()
        #rame = imutils.resize(frame, width=400)
        image = detect_image(net, frame)
        result = np.asarray(image)   
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50, color=(0, 255, 0), thickness=2)
        cv2.namedWindow("result")
	    # show the output frame
        cv2.imshow("result", result)
        #----
        if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
            break
        if cv2.getWindowProperty('result', 0) < 0:  # close windows using close "X" button
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
				
# ----------------------------------
# function: detect_video()
# ----------------------------------
def detect_video( net, video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,4*1000)
    frameRate = cap.get(5)
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
	
    divisor=2
    #fps = "FPS: %d" %  (frameRate)
    frame_count = 0
    while (cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True) :
                break

        t1 = time.time()
		
        if divisor > 0:  # input all the frames
        #if(frameId % divisor == 0):
            image = detect_tracking(net, frame,writeVideo=out)
            result = np.asarray(image)
            fps_imutils.update()
            fps = (fps + (1./(time.time()-t1))) / 2
            fps_txt = "FPS: %f" %  (fps)
            #----Text&result
            cv2.putText(result, text=fps_txt, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 255, 0), thickness=2)
            cv2.namedWindow("result")
            cv2.imshow("result", result)
            #----
            if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
                break
            if cv2.getWindowProperty('result', 0) < 0:  # close windows using close "X" button
                break
				
    cap.release()
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))
    if writeVideo_flag:
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
print("[INFO] process ID: "+idName)

# initialize the output/
outputDir=args.outputDir
checkDir=os.path.exists(outputDir)
if checkDir:
	#removeDir(outputDir)
	ok=0
else:
	os.makedirs(outputDir)
print("[INFO] outputDir: "+outputDir)
output_top_dir = outputDir
output_path = os.path.sep.join([output_top_dir,idName])
if not os.path.exists(output_path):
   os.makedirs(output_path)
	
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

# check out the objects' ID
Court_id=Ball_id=None

for i in range(classesNum) :
    if classes[i] == 'corner':
    	Court_id=i
    if classes[i] == 'ball':
    	Ball_id=i
#print("Court_id={},Ball_id={}".format(Court_id,Ball_id))

# one dnn model
config=args.config
weights=args.weights
	
# read pre-trained model and config file
#net = cv2.dnn.readNet(weights, config)
net = cv2.dnn.readNetFromDarknet(config, weights)

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