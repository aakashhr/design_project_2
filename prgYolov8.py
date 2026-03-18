import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.0.196

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

!mkdir {HOME}/datasets
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3OEqKlaUTLNgrFZGJiv6")
project = rf.workspace("kashish").project("3riders")
version = project.version(2)
dataset = version.download("yolov8")

!yolo task=detect mode=train model=yolov8s.pt data="c:/Users/Haard Shah/Downloads/data.yaml" epochs=100 imgsz=800 plots=True

Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:10]:
    display(Image(filename=image_path, width=600))
    print("\n")
