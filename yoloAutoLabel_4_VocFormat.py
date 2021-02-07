
from imutils.video import VideoStream
import imutils
import cv2
import sys
import os
import argparse
import numpy as np
import random
import colorsys
import re

RESIZED_WIDTH=416
RESIZED_HEIGHT=416

class parser(argparse.ArgumentParser):
	def __init__(self,description):
		super(parser, self).__init__(description)
		self.add_argument(
            "--dataset", "-d", type=str,
            help="path to input datasets (images)",
            metavar="<D>",
        )
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
			"--image", "-i", type=str, required=False,
			help="path to input image",
			metavar="<I>",
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
			help="path to the text file containing class names",
			metavar="<CL>",
		)
		self.add_argument(
			"--confidence", "-cf", type=float, default=0.5, required=False,
			help="minimum probability to filter weak detections [default: %(default)f]",
			metavar="<CF>",
        )
		self.add_argument(
			"--get_img", "-g", type=bool, default=0, required=False,
			help="boolean indicating to output the images/frames",
			metavar="<G>",
        )
		self.add_argument(
			"--use_gpu", "-u", type=bool, default=0, required=False,
			help="boolean indicating if CUDA GPU should be used",
			metavar="<U>",
        )
# ----------------------------------
# class: PascalVOC
# ----------------------------------
class PascalVOC:
   def __init__(self,dir='Annotations'):
      self.init(dir=dir)

   def init(self,dir='Annotations'):
      self.folder=dir
      self.set_filePath(None)
      self.reset()
		
   def reset(self):
      self.objects=[]
      self.width=0
      self.height=0
      self.depth=0
      self.fileTag=''
		
   def new_xml(self,filePath,dir='Annotations'):
      self.reset()
      self.folder=dir
      self.set_filePath(filePath)
		
   def set_filePath(self,filePath):
      self.path=filePath
      if self.path is not None:
         filetemp, fileExtension = os.path.splitext(filePath)
         self.fileTag = os.path.basename(filetemp)
         self.set_fileName(self.fileTag+fileExtension)
		
   def set_fileName(self,fileName):
      self.filename=fileName
		
   def set_size(self,width,height,depth=3):
      self.width=width
      self.height=height
      self.depth=depth
		
   def add_object(self,name,x,y,w,h):
      xmin=x
      ymin=y
      xmax=x+w
      ymax=y+h
      self.objects.append([name,xmin,ymin,xmax,ymax])
		
   def set_top_dir(self,wdir):
      outputDir=os.path.sep.join([wdir,self.folder])
      checkDir=os.path.exists(outputDir)
      if checkDir:
         removeDir(outputDir)
      else:
         os.makedirs(outputDir)
      print("({}) outputDir: {}".format(self.__class__.__name__,outputDir))
      self.top_dir=outputDir

   def object_voc_txt(self,obj,xmlfile):
      name,xmin,ymin,xmax,ymax=obj
      object_name='{}{}{}\n'.format(self.start('name'),name,self.end('name'))
      object_pose='{}{}{}\n'.format(self.start('pose'),'Unspecified',self.end('pose'))
      object_truncated='{}{}{}\n'.format(self.start('truncated'),0,self.end('truncated'))
      object_difficult='{}{}{}\n'.format(self.start('difficult'),0,self.end('difficult'))
      bndbox_xmin='{}{}{}\n'.format(self.start('xmin'),xmin,self.end('xmin'))
      bndbox_ymin='{}{}{}\n'.format(self.start('ymin'),ymin,self.end('ymin'))
      bndbox_xmax='{}{}{}\n'.format(self.start('xmax'),xmax,self.end('xmax'))
      bndbox_ymax='{}{}{}\n'.format(self.start('ymax'),ymax,self.end('ymax'))
      bndbox_str=bndbox_xmin+bndbox_ymin+bndbox_xmax+bndbox_ymax
      bndbox_str='{}\n{}{}\n'.format(self.start('bndbox'),bndbox_str,self.end('bndbox'))
      object_str=object_name+object_pose+object_truncated+object_difficult+bndbox_str
      xmlfile.write('{}{}{}\n'.format(self.start('object'),object_str,self.end('object')))
		
   def output_txt(self):
      xmlfile_path = os.path.sep.join([self.top_dir,self.fileTag+".xml"])
      if os.path.exists(xmlfile_path):
         os.remove(xmlfile_path)
      try:
         xmlfile = open(xmlfile_path, "w")
      except IOError:
         print("({}) [Error] Could not create file {}!!!".format(self.__class__.__name__,xmlfile_path))
         sys.exit()
		
      xmlfile.write('{}\n'.format(self.start('annotation')))
      xmlfile.write('{}{}{}\n'.format(self.start('folder'),self.folder,self.end('folder')))
      xmlfile.write('{}{}{}\n'.format(self.start('filename'),self.filename,self.end('filename')))
      xmlfile.write('{}{}{}\n'.format(self.start('path'),self.path,self.end('path')))
      xmlfile.write('{}{}{}\n'.format(self.start('source'),"<database>Unknown</database>",self.end('source')))
      size_width='{}{}{}\n'.format(self.start('width'),self.width,self.end('width'))
      size_height='{}{}{}\n'.format(self.start('height'),self.height,self.end('height'))
      size_depth='{}{}{}\n'.format(self.start('depth'),self.depth,self.end('depth'))
      size_str=size_width+size_height+size_depth
      xmlfile.write('{}\n{}{}\n'.format(self.start('size'),size_str,self.end('size')))
      xmlfile.write('{}{}{}\n'.format(self.start('segmented'),0,self.end('segmented')))
      for obj in self.objects:
         self.object_voc_txt(obj,xmlfile)
      xmlfile.write('{}\n'.format(self.end('annotation')))
      xmlfile.close()
	
   def end(self,tag):
      return "</"+str(tag)+">"			
   def start(self,tag):
      return "<"+str(tag)+">"
# ----------------------------------
# function: toSort()
# ----------------------------------		
def toSort(imagePaths):
   imagesPath={}
   regex = re.compile(r'_(.*\d+-*\d$)')
   regex2 = re.compile(r'\d')
   for (i, img_path) in enumerate(imagePaths):
         filetemp, fileExtension = os.path.splitext(img_path)
         fileTag = os.path.basename(filetemp)
         x = regex.search(fileTag)
         if x:
            key=fileTag[fileTag.rfind('_'):]
            x = regex2.search(key)
            key=key[x.span()[0]:]
            imagesPath[int(key)]=img_path
   return [imagesPath[k] for k in sorted(imagesPath.keys())]
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
# class: ImageViewLoader
# ----------------------------------
img_formats = ['.png', '.jpeg', '.jpg']
class ImageViewLoader:
   def load(self, imageset, xmlset):
      imagePaths = self.getFilePaths(imageset)
      xmlPaths = self.getFilePaths(xmlset)
      return (imagePaths, xmlPaths)
	  
   def getFilePaths(self, pathset):
      file_paths = []
      for root,directories,files in os.walk(pathset):
         for filename in files:
            file_paths.append(os.path.join(root,filename))
      return file_paths

   def getDirPaths(self, pathset):
      return [d for d in os.listdir(pathset) if os.path.isdir(os.path.join(pathset, d))]
	  
   def generateVoc(self, imagePaths, outputDir, wdir, voc_handler=None):
      for (i, img_path) in enumerate(imagePaths):
         filetemp, fileExtension = os.path.splitext(img_path)
         if fileExtension.lower() not in img_formats:
            #print(fileExtension)
            continue
				
         fileTag = os.path.basename(filetemp)
         image = cv2.imread(img_path)                  
         if voc_handler:
                voc_handler.new_xml(filePath=img_path)
                voc_handler.set_size(width=image.shape[1],height=image.shape[0],depth=3)
				
         image, _ =detect_image(net,image,showOn=0)
         if voc_handler:
                voc_handler.output_txt()
         cv2.namedWindow("result")
         cv2.imshow("result", image)
         if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
                break
         if cv2.getWindowProperty("result", 0) < 0:  # close windows using close "X" button
                break

# ----------------------------------
# function: draw_prediction()
# ----------------------------------
def draw_prediction(img, classname, confidence, x, y, x_plus_w, y_plus_h, color):
    label =  "{}-{:.2f}%".format(classname,confidence * 100)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
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
    scale = 0.00392
	
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
	
    return indices,class_ids,boxes,confidences

# ----------------------------------
# function: detect_image()
# ----------------------------------
def detect_image(net, image, showOn=1):
    global classes,conf_threshold
    indices,class_ids,boxes,confidences = detect_image_by_net(net,image,conf_threshold=conf_threshold)
    # 20200916, to output the PascalVOC format .xml files
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])
        draw_prediction(image, classes[class_ids[i]], confidences[i], x, y, (x + w), (y + h),COLORS[class_ids[i]])
        if voc_handler: voc_handler.add_object(classes[class_ids[i]],x,y,w,h)

    return image, None
# ----------------------------------
# function: detect_camera()
# ----------------------------------
def detect_camera(net):
    width = 1280
    height = 720
    vs = VideoStream(src=0,resolution=(width,height)).start()
    fps = ""
    while True:
        frame = vs.read()
        if voc_handler:
            img_path = saveImg(frame)
            voc_handler.new_xml(filePath=img_path)
            voc_handler.set_size(width=width, height=height, depth=3)
        image, _ = detect_image(net, frame)
        if voc_handler: voc_handler.output_txt()
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
def detect_video( net, video_path,):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC,4*1000)
    frameRate = cap.get(5)
    divisor=2
    #fps = "FPS: %d" %  (frameRate)
    isVideo = max(5,int(frameRate/4))
    width  = round(cap.get(3))
    height = round(cap.get(4))
    while (cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True) :
                break

        if divisor > 0:  # input all the frames
            if voc_handler:
                img_path = saveImg(frame)
                voc_handler.new_xml(filePath=img_path)
                voc_handler.set_size(width=width, height=height, depth=3)
            image, _ = detect_image(net, frame)
            if voc_handler: voc_handler.output_txt()
            result = np.asarray(image)
            cv2.namedWindow("result")
            cv2.imshow("result", result)
            #----
            if cv2.waitKey(1) & 0xFF == 27 :   # [ESC] keys
                break
            if cv2.getWindowProperty('result', 0) < 0:  # close windows using close "X" button
                break
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------
# function: generList()
# ----------------------------------
def generList( net, datasetPath, output_path=None, isSort=0):
    fl = ImageViewLoader()
    dirs=fl.getDirPaths(datasetPath)
    if output_path:
        if voc_handler:
            voc_handler.set_top_dir(output_path)
    if not dirs:
        dirs.append(".")
	
    for dir in dirs:
        #print("dir=",dir)
        cls=os.path.basename(dir)
        if cls == ".":
            cls = "gener"
        src_dir=os.path.sep.join([datasetPath,dir])
        if voc_handler:
            if output_path is None:
                voc_handler.set_top_dir(src_dir)
        imagePaths=fl.getFilePaths(src_dir)
        if isSort: imagePaths=toSort(imagePaths)
        fl.generateVoc(imagePaths, outputDir=output_path, wdir=dir,voc_handler=voc_handler)

def saveImg(img):
    global count
    outputImgDir = outputDir
    if voc_top_dir:
        outputImgDir = voc_top_dir
    filename = "%s/%s%.4d.jpg" % (outputImgDir, voc_imgs_prefix, count)
    if voc_top_dir:
        cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    count = count + 1
    return filename

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
args = parser(description="To generate the Pascal VOC .xml files").parse_args()

files_list=[]
files_list.append([args.classes,"classes"])
files_list.append([args.config,"config"])
files_list.append([args.weights,"weights"])
if args.video:
	files_list.append([args.video,"video"])
if args.image:
	files_list.append([args.image,"image"])
if args.dataset:
	files_list.append([args.dataset,"dataset"])
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
classesNum=len(classes)
# generate different colors for different classes 
COLORS = []
for x in range(classesNum):
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    rgb = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    COLORS.append(rgb)

# one dnn model
config=args.config
weights=args.weights

# read pre-trained model and config file
#net = cv2.dnn.readNet(weights, config)
net = cv2.dnn.readNetFromDarknet(config, weights)
if args.use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

voc_handler=PascalVOC()
voc_top_dir = None
voc_imgs_prefix = "Img"
if args.get_img:
    voc_top_dir = os.path.sep.join([output_path, "IMG"])
    if not os.path.exists(voc_top_dir):
        os.makedirs(voc_top_dir)
count=1
	
isCamera = True
if args.image:
    isCamera = False
    img_path=args.image   
    filetemp, fileExtension = os.path.splitext(img_path)
    fileTag = os.path.basename(filetemp)	
    image = cv2.imread(img_path)
    if voc_handler:
        voc_handler.set_top_dir(output_path)
        voc_handler.new_xml(filePath=img_path)
        voc_handler.set_size(width=image.shape[1], height=image.shape[0], depth=3)
    image,_=detect_image(net,image)
    if voc_handler: voc_handler.output_txt()
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if args.video:
    isCamera = False
    video_url = args.video
    if voc_top_dir:
        filetemp, fileExtension = os.path.splitext(video_url)
        fileTag = os.path.basename(filetemp)
        if len(fileTag) > 10: fileTag=fileTag[0:10]
        voc_imgs_prefix = fileTag
    if voc_handler: voc_handler.set_top_dir(output_path)
    detect_video(net, video_url)

if args.dataset :
    isCamera = False
    datasetPath=args.dataset
    generList(net,datasetPath,output_path=output_path)

if  isCamera :
    if voc_handler: voc_handler.set_top_dir(output_path)
    detect_camera(net)