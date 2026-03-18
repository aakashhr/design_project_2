# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

os.chdir('./models/research')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
print('Done')
os.chdir('../../')

CWD_PATH = '.'

MODEL_RCNN = 'rcnn'
MODEL_YOLO='yolo'

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_RCNN,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_RCNN,'label_map.pbtxt')

configPath="C:/Heron/HelmetRecognitionProposed/HelmetDetection/yolo/yolov3_custom.cfg"
weightsPath="C:/Heron/HelmetRecognitionProposed/HelmetDetection/yolo/yolov3_custom_4000.weights"
labelsPath="C:/Heron/HelmetRecognitionProposed/HelmetDetection/yolo/obj.names"
print('YOLO Path')
print(configPath)
print(weightsPath)
print(labelsPath)

IMAGE_NAME = 'images/sample.jpeg'
VIDEO_NAME = 'videos/clip.mp4'

OUTPUT_FOLDER='output/'

PATH_TO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_FOLDER,'/images')
VIDEO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_FOLDER,'output_clip.mp4')

PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

import tensorflow as tf
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        try:
            od_graph_def.ParseFromString(serialized_graph)
        except Exception as e:
            print(f"Error parsing graph: {e}")
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
print(num_detections)

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
LABELS = open(labelsPath).read().strip().split("\n")

image = cv2.imread(PATH_TO_IMAGE)
image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
image_expanded = np.expand_dims(image, axis=0)

(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

normalizedBoxes = np.squeeze(boxes)
normalizedScores = np.squeeze(scores)
normalizedClasses = np.squeeze(classes)

min_score_thresh = 0.8
detectedBoxes = normalizedBoxes[normalizedScores > min_score_thresh]
detectedClasses = normalizedClasses[normalizedScores > min_score_thresh]

im_height,im_width, _=image.shape
size=(im_width, im_height)

final_boxes = []
for i in range(len(detectedBoxes)):
    ymin, xmin, ymax, xmax = detectedBoxes[i]
    final_boxes.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])

j=0
for [left,right,top,bottom] in final_boxes:

    l=int(round(left))
    r=int(round(right))
    t=int(round(top))
    b=int(round(bottom))
    croppedImage=image[t:b,l:r]

    cv2.rectangle(image, (l,t), (r,b), (255,0,0), 2)

    rows = croppedImage.shape[0]
    cols = croppedImage.shape[1]
    padding=0
    if rows > cols:
        padding = int((rows-cols) / 2)
        paddedImg=cv2.copyMakeBorder(croppedImage, 0, 0, padding, padding,  cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        paddedImg=croppedImage

    (H, W) = paddedImg.shape[:2]
    blob = cv2.dnn.blobFromImage(paddedImg, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxesH = []
    confidencesH = []
    classIDsH = []
    thresholdH = 0.15

    for output in layerOutputs:
        for detection in output:
            confidenceOfEachClass = detection[5:]
            classIDH = np.argmax(confidenceOfEachClass)
            confidenceH = confidenceOfEachClass[classIDH]
            if confidenceH > thresholdH:
                boxH = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, widthH, heightH) = boxH.astype("int")
                x = int(centerX - (widthH / 2))
                y = int(centerY - (heightH / 2))
                boxesH.append([x, y, int(widthH), int(heightH)])
                confidencesH.append(float(confidenceH))
                classIDsH.append(classIDH)
    idxs = cv2.dnn.NMSBoxes(boxesH, confidencesH, thresholdH, 0.1)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxesH[i][0], boxesH[i][1])
            (w, h) = (boxesH[i][2], boxesH[i][3])
            if LABELS[classIDsH[i]] == 'Helmet':
                color = (0, 255, 0)
                cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)
                text = "{}".format(LABELS[classIDsH[i]])
                cv2.putText(croppedImage, text, (x//2, y+ 4*h//3),
                cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            if (LABELS[classIDsH[i]] == 'No Helmet'):
                name=IMAGE_NAME.split(".")[0].split('/')[-1]
                j+=1
                cv2.imwrite( PATH_TO_OUTPUT + '{}.jpg'.format(name+str(j)),croppedImage)

                color = (0, 0, 255)
                cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)
                text = "{}".format(LABELS[classIDsH[i]])
                cv2.putText(croppedImage, text, (x//2, y + 4* h//3),
                cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

video = cv2.VideoCapture(PATH_TO_VIDEO)

images=[]
f=0
while(video.isOpened()):
  if f%3!=0:
    ret,image = video.read()
    f+=1
    continue
  ret,image = video.read()
  if ret == True:
    image=cv2.resize(image,None,fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    normalizedBoxes = np.squeeze(boxes)
    normalizedScores = np.squeeze(scores)
    normalizedClasses = np.squeeze(classes)

    min_score_thresh = 0.8
    detectedBoxes = normalizedBoxes[normalizedScores > min_score_thresh]
    detectedClasses= normalizedClasses[normalizedScores > min_score_thresh]

    im_height,im_width, _=image.shape
    size=(im_width, im_height)

    final_boxes = []
    for i in range(len(detectedBoxes)):
        ymin, xmin, ymax, xmax = detectedBoxes[i]
        final_boxes.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])

    j=0
    for [left,right,top,bottom] in final_boxes:
        l=int(round(left))
        r=int(round(right))
        t=int(round(top))
        b=int(round(bottom))
        croppedImage=image[t:b,l:r]

        cv2.rectangle(image, (l,t), (r,b), (255,0,0), 2)

        rows = croppedImage.shape[0]
        cols = croppedImage.shape[1]
        padding=0
        if rows > cols:
            padding = int((rows-cols) / 2)
            paddedImg=cv2.copyMakeBorder(croppedImage, 0,0,padding, padding,  cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            paddedImg=croppedImage

        (H, W) = paddedImg.shape[:2]
        blob = cv2.dnn.blobFromImage(paddedImg, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxesH = []
        confidencesH = []
        classIDsH = []
        thresholdH = 0.15

        for output in layerOutputs:
            for detection in output:
                confidenceOfEachClass = detection[5:]
                classIDH = np.argmax(confidenceOfEachClass)
                confidenceH = confidenceOfEachClass[classIDH]
                if confidenceH > thresholdH:
                    boxH = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, widthH, heightH) = boxH.astype("int")
                    x = int(centerX - (widthH / 2))
                    y = int(centerY - (heightH / 2))
                    boxesH.append([x, y, int(widthH), int(heightH)])
                    confidencesH.append(float(confidenceH))
                    classIDsH.append(classIDH)
        idxs = cv2.dnn.NMSBoxes(boxesH, confidencesH, thresholdH, 0.1)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxesH[i][0], boxesH[i][1])
                (w, h) = (boxesH[i][2], boxesH[i][3])
                if LABELS[classIDsH[i]] == 'Helmet':
                    color = (0, 255, 0)
                    cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)
                    text = "{}".format(LABELS[classIDsH[i]])
                    cv2.putText(croppedImage, text, (x//2, y+ 4*h//3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                if (LABELS[classIDsH[i]] == 'No Helmet'):
                    name=VIDEO_NAME.split(".")[0].split('/')[-1]+"_"+str(f)+"_"
                    j+=1
                    cv2.imwrite(PATH_TO_OUTPUT + '{}.jpg'.format(name+str(j)),croppedImage)
                    color = (0, 0, 255)
                    cv2.rectangle(croppedImage, (x-padding, y), (x + w-padding, y + h), color, 2)
                    text = "{}".format(LABELS[classIDsH[i]])
                    cv2.putText(croppedImage, text, (x//2, y + 4* h//3),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        images.append(image)
    f+=1
  else:
    break
video.release()

out = cv2.VideoWriter(VIDEO_OUTPUT,cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
for i in range(len(images)):
    out.write(images[i])
out.release()
