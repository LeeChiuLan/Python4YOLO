# -*- coding: utf-8 -*-
"""
this is base on the code created by @author: Arun Ponnusamy


"""
#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


from imutils.video import VideoStream
import imutils
import sys
import os
import cv2
import argparse
import numpy as np
import argparse
import time

class parser(argparse.ArgumentParser):
	def __init__(self,description):
		super(parser, self).__init__(description)
		self.add_argument(
            "--outputDir", "-o",default='output', type=str,
            help="path to the output directory of list files",
            metavar="<O>",
        )
		self.add_argument(
            "--image", "-i", type=str, required=False,
            help="path to input image",
            metavar="<I>",
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
			"--save", "-s", type=bool, default=0, required=False,
			help="boolean indicating to save the output images or the testing video",
			metavar="<S>",
        )
		self.add_argument(
			"--use_gpu", "-u", type=bool, default=0, required=False,
			help="boolean indicating if CUDA GPU should be used",
			metavar="<U>",
        )
		self.add_argument(
			"--dont_show", "-dont", type=bool, default=0, required=False,
			help="boolean indicating not to display any pictures",
			metavar="<DONT>",
        )
# ----------------------------------
# class: SaveImage
# ----------------------------------
class SaveImage:
    def __init__(self):  # format: 'xyrb' or 'xywh'
    	self.top_dir=None
    	
    def set_filePath(self,filePath):
    	self.path=filePath
    	if self.path is not None:
            filetemp, fileExtension = os.path.splitext(filePath)
            self.fileTag = os.path.basename(filetemp)
            self.set_fileName(self.fileTag+".jpg")
			
    def set_top_dir(self,outputDir):
    	checkDir=os.path.exists(outputDir)
    	if checkDir:
            #removeDir(outputDir)
            ok=0
    	else:
            os.makedirs(outputDir)
    	self.top_dir=outputDir

    def set_fileName(self,fileName):
    	self.filename=fileName

    def output(self,image=None):
    	if image is None:
            return
    	print("file: "+self.filename)
    	self.file_path = os.path.sep.join([self.top_dir,self.fileTag+".jpg"])
    	cv2.imwrite(self.file_path, image,[cv2.IMWRITE_JPEG_QUALITY, 90])

    def output_result_image(self,image=None):
    	if image is None:
            return
    	self.file_path = os.path.sep.join([self.top_dir,self.fileTag+".result.jpg"])
    	cv2.imwrite(self.file_path, image,[cv2.IMWRITE_JPEG_QUALITY, 90])

    def display(self):
    	print("({}) outputDir: {}".format(self.__class__.__name__,self.top_dir))
# ----------------------------------
# ----------------------------------
# function: detect_camera()
# ----------------------------------
def detect_camera(net, resolution=416):
    vs = VideoStream(src=0).start()
    fps = ""
    while True:
        frame = vs.read()
        #rame = imutils.resize(frame, width=400)
        image = detect_image(net, frame, resolution=resolution)
        result = np.asarray(image)   
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50, color=(0, 255, 0), thickness=2)
        
	    # show the output frame
        if to_show:
            cv2.namedWindow("result")
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
def detect_video( net, video_path, resolution=416):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,4*1000)
    frameRate = cap.get(5)
    out = None
    if writeVideo_flag:
        filetemp, fileExtension = os.path.splitext(video_path)
        fileTag = os.path.basename(filetemp)
        w = int(cap.get(3))
        h = int(cap.get(4))
        out, output_video_path = output_video(output_video_dir,w, h,fileTag=fileTag)
        
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret != True) :
                break

        t1 = time.time()
		
        image = detect_image(net, frame, resolution=resolution)
        result = np.asarray(image)
        fps_imutils.update()
        fps = (fps + (1./(time.time()-t1))) / 2
        fps_txt = "FPS: %f" %  (fps)
        #----Text&result
        cv2.putText(result, text=fps_txt, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=0.50, color=(0, 255, 0), thickness=2)

        if out: out.write(result)
        if to_show:
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
        print("[INFO] output video : "+output_video_path)
    cv2.destroyAllWindows()

# ----------------------------------
# function: get_output_layers()
# ----------------------------------
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# ----------------------------------
# function: draw_prediction()
# ----------------------------------
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    #label = str(classes[class_id])
    label =  "{}-{:.2f}%".format(classes[class_id],confidence * 100)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ----------------------------------
# function: detect_image()
# ----------------------------------
def detect_image(net, image, resolution=416):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
	
    inpWidth=inpHeight=resolution
	
	# create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (inpWidth, inpHeight), (0,0,0), True, crop=False)
	
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

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
		
    return image
# ----------------------------------
# function: output_video()
# ----------------------------------
def output_video(output_path,width, height,fileTag='output'):
	video_name="{}.avi".format(fileTag)
	output_video_path = os.path.sep.join([output_path,video_name])
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
	return out, output_video_path	

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
RESIZED_WIDTH=416
RESIZED_HEIGHT=416
args = parser(description="yolo detection").parse_args()
files_list=[]
files_list.append([args.classes,"classes"])
files_list.append([args.config,"config"])
files_list.append([args.weights,"weights"])
if args.image:
	files_list.append([args.image,"image"])
if args.video:
	files_list.append([args.video,"video"])
files_exist_check(files_list)

# show information on the process ID
myProcessID=os.getpid()
idName='{}_{}'.format("PID", myProcessID)
print("[INFO] process ID: "+idName)

# save the output images,video 
save_handler=None
writeVideo_flag = False
if args.save:
    # initialize the output/
    outputDir=args.outputDir
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    output_top_dir = outputDir
    save_handler=SaveImage()

# 
to_show=1
if args.dont_show==1:
    to_show=None


nms_threshold = 0.4
conf_threshold = args.confidence;
# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

if args.use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

isCamera = True
if args.image:
    isCamera = False
    if save_handler:
    	output_images_path = os.path.sep.join([output_top_dir,idName+"_IMG"])
    	save_handler.set_top_dir(output_images_path)
    img_path=args.image
    image = cv2.imread(img_path)
    image=detect_image(net,image,resolution=RESIZED_WIDTH)
    if save_handler:
        save_handler.set_filePath(img_path)
        save_handler.output_result_image(image)
        save_handler.display()
    if to_show: cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if args.video:
    isCamera = False
    if save_handler:
    	writeVideo_flag = True
    	output_video_dir = os.path.sep.join([output_top_dir,idName+"_Video"])
    	if not os.path.exists(output_video_dir):
    	    os.makedirs(output_video_dir)
    video_url = args.video
    detect_video(net, video_url,resolution=RESIZED_WIDTH)

if  isCamera :
    detect_camera(net,resolution=RESIZED_WIDTH)