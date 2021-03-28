# YOLOv4 + Deep SORT with OpenCV.dnn - [Michael Jackson Tribute](https://youtu.be/t3JBsGrZIpA?list=RDW8QG4wg2Pms)
The raw video from Multiple Object Tracking Benchmark - MOT17-02-FRCNN [[link](https://motchallenge.net/vis/MOT17-02-FRCNN)]

#### [Demo] YOLOv4-Tiny + Deep SORT,  GPU-1050Ti CUDA Enabled
[![Alt text](yolov4-deep-sort-Opencv.dnn.gif)](https://youtu.be/mQXgsk38I7w)

## yolo_deep_sort_main.py
<div>
  usage: Detection and Tracking with OpenCV.dnn (YOLOv4 + Deep SORT)
  
  [-h] [--outputDir <O>] [--video <V>] --config
                                              <C> --weights <W> --classes <CL>
                                              [--confidence <CF>] [--use_gpu <U>]
</div>
<div>
optional arguments:
<ul>
  <li>
  -h, --help            show this help message and exit</li>

  <li>--outputDir <O>, -o <O>
                        path to the output directory of list files</li>
  <li>--video <V>, -v <V>   path to input Video</li>
  <li>--config <C>, -c <C>  path to yolo config file</li>
  <li>--weights <W>, -w <W>
                        path to yolo pre-trained weights</li>
  <li>--classes <CL>, -cl <CL>
                        path to text file containing class names</li>
  <li>--confidence <CF>, -cf <CF>
                        minimum probability to filter weak detections [default: 0.5]</li>
  <li>--use_gpu <U>, -u <U>
                        boolean indicating if CUDA GPU should be used</li>


</ul>         
</div>

## Setup

#### 1. Requirements

```bash
python3
Tensorflow
OpenCV (version >= 4.4.0 suporting YOLOv4)
```
The others packages should be installed according to the messages shown while executing this program 

#### 2. Deep SORT

reference to [theAIGuysCode/yolov3_deepsort](https://github.com/theAIGuysCode/yolov3_deepsort.git)
```bash
# change directory to tracking
cd tracking

git clone https://github.com/theAIGuysCode/yolov3_deepsort.git
```
#### 3. YOLOv4

You'd prepare your trained model by weights, config file, lasses names.

In this article we will use those files from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet): yolov4.weights, yolov4.cfg, coco.names

## How to execute program

#### Command line

```bash
python3 yolo_deep_sort_main.py --weights yolov4.weights --config yolov4.cfg --classes coco.names --video MOT17-02-FRCNN-raw.webm
```
if test with yolov4-tiny
```bash
python3 yolo_deep_sort_main.py --weights yolov4-tiny.weights --config yolov4-tiny.cfg --classes coco.names --video MOT17-02-FRCNN-raw.webm
```

#### Abort program
As the program is running , to click the video window then press the [ESC] key to stop the program.

#### Auto output the results
You can find the output results under the folder 'outputDir' with its own PID there.

To disable this function by setting 'writeVideo_flag = False'  in Line #387.


#### Checking

```bash
if got error in 'net/images:0'
```
![](net-error-00.png)
```bash
To modify "net/%s:0" => "%s:0" 83 & 85 lines in 'yolov3_deepsort/tools/generate_detections.py'
```

#### Run with CUDA

If your environment is ready with NVIDIA GPUs CUDA also the OpenCV, then make speed faster  by add '-u 1'.

reference to [How to use OpenCV’s “dnn” module with NVIDIA GPUs, CUDA, and cuDNN](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) by Adrian Rosebrock.
(Note: change the OpenCV’s version to 4.4.0)
```bash
# option '--use_gpu, -u'
python3 yolo_deep_sort_main.py --weights yolov4.weights --config yolov4.cfg --classes coco.names --video MOT17-02-FRCNN-raw.webm -u 1
```
#### Show the detections

change 'Show_Detect=1' in Line #132 as below:
```bash
def detect_tracking(net, image, Show_Detect=1):
```
#### Run with Camera

if none of the given videos it will be run with camera. 