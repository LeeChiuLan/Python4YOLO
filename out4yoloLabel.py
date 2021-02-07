# -*- coding: utf-8 -*-
"""
this is base on the code created by @author: Guanghan Ning


"""
import xml.etree.ElementTree as ET
import os
from os import walk, getcwd
from PIL import Image
import argparse
import sys
import cv2

img_formats = ['.png', '.jpeg', '.jpg']
# ----------------------------------
# class: ImageViewLoader
# ----------------------------------
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
	  
   def generateYOLOLabels(self, imagePaths, xmlDir, outputDir, list_file, mergein=0, verbose=-1):
      for (i, img_path) in enumerate(imagePaths):
         filetemp, fileExtension = os.path.splitext(img_path)
         if fileExtension.lower() not in img_formats:
            #print(fileExtension)
            continue
         fileTag = os.path.basename(filetemp)
         spec_dir=outputDir
         if mergein == 1:
            spec_dir=os.path.dirname(img_path)
         txt_outpath= os.path.sep.join([spec_dir,fileTag+".txt"])
         
         if xmlDir is None:
            if border is not None:
               img_path_ret=takefromtExtImg(img_path,txt_outpath,fileTag)
               if img_path_ret is not None:
                  img_path=img_path_ret
            else:
               takefromtImg(img_path,txt_outpath)
         else:
            xml_path= os.path.sep.join([xmlDir,fileTag+".xml"])
            takefromtXml(xml_path,txt_outpath)

         if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
               len(imagePaths)))

         list_file.write('%s\n'%(img_path))
# ----------------------------------
# function: toSort()
# ----------------------------------		
def toSort(imagePaths, ontxt=False):
   print("size 1={}".format(len(imagePaths)))
   imagesPath={}
   for (i, img_path) in enumerate(imagePaths):
         filetemp, fileExtension = os.path.splitext(img_path)
         if fileExtension.lower() not in img_formats:
            continue
         fileTag = os.path.basename(filetemp)
         if ontxt: fileTag=fileTag[:fileTag.find('_')]
         if fileTag.isdigit():
            key=fileTag
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
# function: convert()
# ----------------------------------
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
# ----------------------------------
# function: takefromtXml()
# ----------------------------------
def takefromtXml(xml_path,txt_outpath):
   txt_outfile = open(txt_outpath, "w")
   if not os.path.exists(xml_path):
      return
   in_file = open(xml_path)
   tree=ET.parse(in_file)
   root = tree.getroot()
   size = root.find('size')
   w = int(size.find('width').text)
   h = int(size.find('height').text)
   
   for obj in root.iter('object'):
      difficult = obj.find('difficult').text
      cls = obj.find('name').text
      if cls not in label_names or int(difficult) == 1:
         continue
      cls_id = index_of(cls,label_names)
      xmlbox = obj.find('bndbox')
      box = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
      size=(w,h)
      bb = convert(size, box)
      txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
      counts_dic[cls_id]=counts_dic[cls_id]+1      #counts++ for this label/class

   txt_outfile.close()
# ----------------------------------
# function: takefromtImg()
# ----------------------------------
def takefromtImg(img_path,txt_outpath):
   global tabin
   
   if cls_id < 0:
      print("[Error] unable to classiffy the label name !!!")
      return
   txt_outfile = open(txt_outpath, "w")
   im=Image.open(img_path)
   w= int(im.size[0])
   h= int(im.size[1])
   xmin = 0
   ymin = 0
   xmax = w
   ymax = h
   
   # retracted by p(pixel) on each bndbox
   p = tabin
   w = w - 2*p
   h = h - 2*p
   xmin = xmin + p
   ymin = ymin + p
   xmax = xmin + w
   ymax = ymin + h
         
   box = (float(xmin), float(xmax), float(ymin), float(ymax))
   size=(w,h)
   bb = convert(size, box)
   txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
   counts_dic[cls_id]=counts_dic[cls_id]+1      #counts++ for this label/class
   txt_outfile.close()
# ----------------------------------
# function: takefromtExtImg()   /* the expanded image with the new border */ 
# ----------------------------------
def takefromtExtImg(img_path,txt_outpath,fileTag):
   global border
   global outputDir
   global mergein
   global tabin
   global bRGB
   
   if cls_id < 0:
      print("[Error] unable to classiffy the label name !!!")
      return None

   saveDir= os.path.sep.join([outputDir,label_names[cls_id]])
   if not os.path.exists(saveDir):
      os.makedirs(saveDir)

   new_txt_outpath = txt_outpath
   newImg_path = os.path.sep.join([saveDir, fileTag+".jpg"])
   newImg_path = os.path.abspath(newImg_path)
   if mergein == 1:
      new_txt_outpath = os.path.sep.join([saveDir, fileTag+".txt"])
   txt_outfile = open(new_txt_outpath, "w")
   cv2image = cv2.imread(img_path)
   height, width, channels = cv2image.shape
   xmin = 0
   ymin = 0
   xmax = w = width
   ymax = h = height
   if(width < border) or (height < border):
      color=[bRGB[2], bRGB[1], bRGB[0]]      #RGB to BGR   // to OpenCV
      w=max(border,width)
      h=max(border,height)
      left=int((w-width)/2)
      right=left
      top=int((h-height)/2)
      bottom=top
      cv2image=cv2.copyMakeBorder(cv2image,top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      xmin = xmin + left
      ymin = ymin + top
      xmax = xmin + width
      ymax = ymin + height

   # retracted by p(pixel) on each bndbox
   p = tabin
   w = w - 2*p
   h = h - 2*p
   xmin = xmin + p
   ymin = ymin + p
   xmax = xmin + w
   ymax = ymin + h
   
   box = (float(xmin), float(xmax), float(ymin), float(ymax))
   size=(w,h)
   bb = convert(size, box)
   txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
   counts_dic[cls_id]=counts_dic[cls_id]+1      #counts++ for this label/class
   txt_outfile.close()
   cv2.imwrite(newImg_path,cv2image)
   return newImg_path
   
# ----------------------------------
# function: index_of()
# ----------------------------------
def index_of(val, in_list):
    try:
        return in_list.index(val)
    except ValueError:
        return -1 
# ----------------------------------
# function: PrintLog()
# ----------------------------------
def PrintLog(message=''):
   global logfile
   print(message)
   logfile.write(message+"\n")
   logfile.flush()

# ----------------------------------
# function: PrintSystemInfo()
# ----------------------------------
def PrintSystemInfo(mode='log_only'):
   global datasetPath
   global labelPath
   global xmlset
   global label_names
   global classes_len
   global border
   global outputDir

   global logfile
   
   title=''.join(["|-------------------------------|\n",
                  "|  Sytem information   |\n",
                  "|-------------------------------|\n"])
   
   label_classes = ','.join(label_names)
   message = ''.join([
                     title,
                     "dataset                    : "+datasetPath+"\n",
                     "label text file            : "+labelPath+"\n",
                     "Log file                    : "+logfile_path+"\n",
                     "classes lables("+str(classes_len)+")= ",label_classes+"\n"])
   if xmlset is not None:
      message=message+"xmlset                    : "+xmlset+"\n"
   if border is not None:
      message=message+"border  {}: to expand the image size {}x{}) \n".format(bRGB,border,border)
      if tabin is not None and tabin > 0:
         message=message+"tabin : retracted by {} pixcels on each bndbox\n".format(tabin)
   if outputDir is not None:
      message=message+"outputDir                 : "+os.path.abspath(outputDir)+"\n"

   message=message+"\n"
   logfile.write(message)
   if not mode == 'log_only':
      print(message)
   


"""-------------------------------------------------------------------""" 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
   help="path to input datasets")
ap.add_argument("-x", "--xmlset", required=False,
   help="path to the folder for the input-xml-files")
ap.add_argument("-l", "--label", required=False,
   help="path to the label(text file)")
ap.add_argument("-m", "--mergein", required=False,
   help="yes:: put the output files under the same path as their images are")
ap.add_argument("-b", "--border", required=False,
   help="500:: make border on each image to get the new image with 500x500")
ap.add_argument("-c", "--bColor", required=False,
   help="255:255:255: the border's color (RGB)")
ap.add_argument("-t", "--tabin", required=False,
   help="the value of pixcel: retracted by this value on each bndbox")
ap.add_argument("-o", "--outputDir", required=False,
   help="path to output directory")
ap.add_argument("-v", "--verbose", required=False,
   help="0: no interactive")
args = vars(ap.parse_args())
if len(sys.argv) < 2:
   print("no argument")
   sys.exit()

interactive=1
if ap.parse_args().verbose:
   interactive =int(args["verbose"]) 

mergein=0
if ap.parse_args().mergein:
   if args["mergein"] == "yes":
      mergein = 1

# show information on the process ID
myProcessID=os.getpid()
idName='{}_{}'.format("PID", myProcessID)
print("[INFO] process ID: "+idName)

# initialize the variables
datasetPath=None
labelPath=''
xmlset=None
border=None
bColor="255:255:255"
bRGB=None
tabin=None
outputDir=None

# initialize the class labels
label_names=["ball", "carolinamarin", "racket1","racket2", "taitzuying"] #default
if ap.parse_args().label:
   labelPath=args["label"]
   if os.path.exists(labelPath):
      label_file = open(labelPath)
      label_names=label_file.read().split(',')
      label_names=label_names[:-1]
classes_len=len(label_names)
print("[INFO] classes lables("+str(classes_len)+")=",label_names)

fl = ImageViewLoader()

if ap.parse_args().xmlset:
   xmlset = args["xmlset"]
   print("[INFO] xmlset : ",xmlset)

if ap.parse_args().dataset:
   datasetPath = args["dataset"]
if datasetPath is None:
   print("[Error] \"dataset\" not given!!!")
   sys.exit()
if not os.path.exists(datasetPath):
   print("[Error] dataset: \""+datasetPath+"\" not found !!!")
   sys.exit()
if not os.listdir(datasetPath):
   print("[Error] dataset: \""+datasetPath+"\" is empty !!!")
   sys.exit()
print("[INFO] dataset : ", datasetPath)

if ap.parse_args().bColor:
   bColor = args["bColor"]

if ap.parse_args().tabin:
   tabin = int(args["tabin"])

if ap.parse_args().border:
   border = int(args["border"])
   if border < 256:
      print("[Warning] border : {} too small. suggestion more than 256")
   print("[INFO] border : yes (to expand the image size ({}x{}) ".format(border,border))
   outputDir="imagesDir"
   bRGB= [int(i) for i in bColor.split(':')]
   print("[INFO] bColor : "+str(bRGB)+" RGB the border's color ")
   
if tabin is None:
   tabin=0
else:
   print("[INFO] tabin : retracted by {} pixcels on each bndbox".format(tabin))
 
if ap.parse_args().outputDir:
   outputDir = args["outputDir"]
if outputDir is not None:
   outputDir=os.path.sep.join([outputDir,idName])
   if not os.path.exists(outputDir):
      os.makedirs(outputDir)
   print("[INFO] outputDir : ",os.path.abspath(outputDir))
# ===================================================
if interactive == 1:
   if sys.version_info[0] < 3:
      key = raw_input('All correct? [Y/N] ')      # python 2
   else:
      key = input('All correct? [Y/N] ')
   if key[0] == 'n':
      sys.exit()
# ===================================================


# initialize the output/
output_top_dir = "output"
output_top_dir = os.path.sep.join([output_top_dir,'yoloLabels'])
output_path = os.path.sep.join([output_top_dir,idName])
if not os.path.exists(output_path):
      os.makedirs(output_path)
   
# initialize the log
logfile_path = os.path.sep.join([output_path,idName+".log"])
if os.path.exists(logfile_path):
   os.remove(logfile_path)
try:
   logfile = open(logfile_path, "w")
except IOError:
   print("[Error] Could not create file \""+logfile_path+"\"!!!")
   sys.exit()

PrintSystemInfo(mode='log_only')
   
# initialize the list for the counts of pictures(be annotated) by each cleass
counts_dic={}
for i in label_names:
   t=index_of(i,label_names)
   counts_dic[t]=0

dirs=fl.getDirPaths(datasetPath)
if not dirs:
   #list_file = open('%s/yolo_list.txt'%(output_path), 'w')
   dirs.append(".")

for dir in dirs:
   #print("dir=",dir)
   cls=os.path.basename(dir)
   cls_id=index_of(cls,label_names)
   if cls_id<0:
      PrintLog(message="directoriy \"{}\" not the label name for the classes".format(cls))
      #continue
   if cls == ".":
      cls = "myYolo"
   list_file = open('%s/%s_list.txt'%(output_path, cls), 'w')
   output_dir = os.path.sep.join([output_path,dir])
   if not os.path.exists(output_dir) and mergein == 0:
      os.makedirs(output_dir)
   imagePaths=fl.getFilePaths(os.path.sep.join([datasetPath,dir]))
   imagePaths=toSort(imagePaths)
   print("size 2={}".format(len(imagePaths)))
   fl.generateYOLOLabels(imagePaths, xmlDir=xmlset, outputDir=output_dir, list_file=list_file, mergein=mergein, verbose=500)

   list_file.close()
 
# list the results
PrintLog(message="\n=============<<<  Result  >>>=============")
PrintLog(message="[ %YOLO format Annotations% ] unit: items")
for i in label_names:
   t=index_of(i,label_names)
   message="   {} : {}".format(i,counts_dic[t])
   PrintLog(message=message)
PrintLog(message="\n======================================")
print("[INFO] log : ", os.path.abspath(logfile_path))