# Python4YOLO
Utilities for YOLOv4


Lee, Chiu-Lan(李秋蘭)  
<br>
My Website: <a href="https://leechiulan.github.io" target="_blank">https://leechiulan.github.io</a><br>
<br><br>
My GitHub : <a href="https://github.com/LeeChiuLan" target="_blank">https://github.com/LeeChiuLan</a><br>
<br><br>
My Thesis : Ball Detection and Tracking in Badminton Game Videos Based on Deep Learning->[Demo](https://youtu.be/HJgLzsmGjpk)
    

## Programs
- out4yoloLabel.py
  - 
- yoloAutoLabel_4_VocFormat.py
  - To generate the Pascal VOC .xml files
- yolo_deep_sort_main.py
  - Detection and Tracking with OpenCV.dnn
- yolo_detector.py
  - yolo detection
  
<hr> 
## yolo_detector.py
<div>
  usage: yolo_detector.py 
  
  [-h] [--outputDir <O>] [--image <I>] [--video <V>]
                      --config <C> --weights <W> --classes <CL>
                      [--confidence <CF>] [--save <S>] [--use_gpu <U>]
                      [--dont_show <DONT>]
</div>
<div>
optional arguments:
<ul>
  <li>
  -h, --help            show this help message and exit</li>

  <li>--outputDir <O>, -o <O><br>
                        path to the output directory of list files</li>
  <li>--image <I>, -i <I>   path to input image</li>
  <li>--video <V>, -v <V>   path to input Video</li>
  <li>--config <C>, -c <C>  path to yolo config file</li>
  <li>--weights <W>, -w <W><br>
                        path to yolo pre-trained weights</li>
  <li>--classes <CL>, -cl <CL><br>
                        path to text file containing class names</li>
  <li>--confidence <CF>, -cf <CF><br>
                        minimum probability to filter weak detections</li>
                        [default: 0.500000]</li>
  <li>--save <S>, -s <S>    boolean indicating to save the output images or the<br>
                        testing video</li>
  <li>--use_gpu <U>, -u <U><br>
                        boolean indicating if CUDA GPU should be used</li>
  <li>--dont_show <DONT>, -dont <DONT><br>
                        boolean indicating not to display any pictures</li>

</ul>         
</div>
