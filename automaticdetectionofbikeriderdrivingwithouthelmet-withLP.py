# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import pytesseract  # For OCR

# Set Tesseract OCR executable path (Update the path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Paths
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'video1.mp4'  # Video file
CWD_PATH = os.getcwd()

# Paths to the helmet and license plate detection models
PATH_TO_HELMET_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LP_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'license_plate_frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 4  # Update according to your dataset

# Load label maps for helmet detection
from object_detection.utils import label_map_util, visualization_utils as vis_util

helmet_label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
helmet_categories = label_map_util.convert_label_map_to_categories(helmet_label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
helmet_category_index = label_map_util.create_category_index(helmet_categories)
import sys

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

    
def send_mail():#water,land):
    
 # Email sender and receiver configuration
 sender_email = "softproms301@gmail.com"#"softproms14@gmail.com"  # Replace with your email address
 sender_password ="nwimmyogblpcoxml"# "lpkblymuyqducryv"      # Replace with your email password (use app password for Gmail)
 receiver_email = "softproms2@gmail.com"  # Replace with recipient's email address

 # Email subject and body
 subject = "Helmet Alert"
 body = "An Image without Helmet is found out!"

 # Create email object
 message = MIMEMultipart()
 message["From"] = sender_email
 message["To"] = receiver_email
 message["Subject"] = subject

 # Attach email body
 message.attach(MIMEText(body, "plain"))


 try:
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Start server connection
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Upgrade connection to secure encrypted SSL/TLS

    # Log in to the email account
    server.login(sender_email, sender_password)
    
    # Send the email
    server.sendmail(sender_email, receiver_email, message.as_string())
    print("Email sent successfully!")

 except Exception as e:
    print(f"Failed to send email: {e}")

 finally:
    server.quit()

# Load TensorFlow models into memory
def load_model(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')
    return detection_graph

helmet_graph = load_model(PATH_TO_HELMET_CKPT)
lp_graph = load_model(PATH_TO_LP_CKPT)

# Start video capture
video = cv2.VideoCapture(os.path.join(CWD_PATH, VIDEO_NAME))

with helmet_graph.as_default():
    helmet_sess = tf.compat.v1.Session(graph=helmet_graph)
    with lp_graph.as_default():
        lp_sess = tf.compat.v1.Session(graph=lp_graph)

# Define tensors for helmet detection
image_tensor = helmet_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = helmet_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = helmet_graph.get_tensor_by_name('detection_scores:0')
detection_classes = helmet_graph.get_tensor_by_name('detection_classes:0')
num_detections = helmet_graph.get_tensor_by_name('num_detections:0')

# Define tensors for license plate detection
lp_image_tensor = lp_graph.get_tensor_by_name('image_tensor:0')
lp_detection_boxes = lp_graph.get_tensor_by_name('detection_boxes:0')
lp_detection_scores = lp_graph.get_tensor_by_name('detection_scores:0')
lp_detection_classes = lp_graph.get_tensor_by_name('detection_classes:0')
lp_num_detections = lp_graph.get_tensor_by_name('num_detections:0')
mailsent=0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Helmet detection
    (boxes, scores, classes, num) = helmet_sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    if (num>0 and mailsent==0):
       send_mail()
       mailsent=1
    print(num)
    #print('Classes:', str(classes))
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        helmet_category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # License plate detection
    (lp_boxes, lp_scores, lp_classes, lp_num) = lp_sess.run(
        [lp_detection_boxes, lp_detection_scores, lp_detection_classes, lp_num_detections],
        feed_dict={lp_image_tensor: frame_expanded})

    for i in range(int(lp_num[0])):
        if lp_scores[0][i] > 0.60:  # Confidence threshold
            ymin, xmin, ymax, xmax = lp_boxes[0][i]
            (im_height, im_width) = frame.shape[:2]
            (xmin, xmax, ymin, ymax) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
            license_plate_roi = frame[ymin:ymax, xmin:xmax]

            # Perform OCR on the cropped license plate
            license_text = pytesseract.image_to_string(license_plate_roi, config='--psm 7')
            cv2.putText(frame, license_text.strip(), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw bounding box for license plate
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            if mailsent==0:
              send_mail()  
    cv2.imshow('Helmet and License Plate Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
