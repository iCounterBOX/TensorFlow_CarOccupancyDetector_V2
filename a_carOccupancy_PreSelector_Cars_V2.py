'''
 OBJECT DETECTION -   local  V I D E O     &  local cocoInceptionModel  
 CKOSS / 28.08.19 / 20:34
 
              
 
 From a MP4 Video we predict CARS and write those single CAR-Pictures to a Folder
 OFFICIAL TUTORIAL and source
 - https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html


 study:
 take picture from video:   https://www.geeksforgeeks.org/extract-images-from-video-in-python/

 How To Train an Object Detection Classifier Using TensorFlow 1.5 (GPU) on Windows 10
 https://www.youtube.com/watch?v=Rgpfk6eYxJA
 
 ToDo:
     If ssd_inception_v2_coco_2017_11_17.tar.gz exist on hdd  skip the download!!

'''

# ACHTUNG: WIR brauchen hier diese  visualization_utils_ck4Cars.pc
# von hier: "C:\appl\TensorFlow\models\research\object_detection\utils"  

import platform
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2   # opencv

import sys
sys.path.append("C:\\appl\\TensorFlow\\models\\research\\")
sys.path.append("C:\\appl\\TensorFlow\\models\\research\\object_detection\\utils")

from utils import label_map_util
from utils import visualization_utils_ck4CarsSinglePic as vis_util
#from utils import visualization_utils as vis_util

print("V E R S I O - I N F O:")
print("OpenCV Version: {}".format(cv2.__version__))
print("Python Version: " + platform.python_version())
print("Numpy Version: " +  np.__version__)
print("TensorFlow Version: " +  tf.__version__)



# Define the video stream
cap = cv2.VideoCapture('./video/carsInFront.mp4')

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90


# ckoss ---------------------------------------------------------
# Download Model
print("Download Model - Start")

path = './' + MODEL_FILE
if os.path.isfile(path) and os.access(path, os.R_OK):
    print ("File exists and is readable ( REGULAR PROCESSING ) / file: " + path)
else:
    print ("File: " + path + " NOT existing - Downloading this from URL: " + DOWNLOAD_BASE + MODEL_FILE)
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)


# ckoss / END  ---------------------------------------------------


# ckoss / process MODEL 
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
print("Download Model - END")

# Load a (frozen) Tensorflow model into memory.

print("Download frozen Model - Start")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("Download frozen Model - END")

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

frames_counter = 1

# Detection
print("detection_graph.as_default() - Start")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            frames_counter = frames_counter + 1            
            
            ret, frame = cap.read()
            #print(frame)
            #print(check)
            if ret:
                #gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
               # cv2.imshow("Capturing", gray)
                
                # prediction-Loop
                
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                frame_expanded = np.expand_dims(frame, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                
                
                # Here output the category as string and score to terminal
                #print([category_index.get(i) for i in classes[0]])
                #print(scores)                
                
                
                
                # Visualization of the results of a detection. - search for ckoss visualization_utils.py
                crop_img = vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1,       # thickness original is 8    
                    WriteSinglePic2File=True)  # CKOSS  - Write the single image to a CAR-File?                                 
    
                # Display output
                cv2.imshow('object detection', cv2.resize(frame, (960, 720)))
                if crop_img is not None:
                  cv2.imshow('Single CAR for Prediction', cv2.resize(crop_img, (960, 720)))     # ckoss
                
                #EXIT CHECKER
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                        cv2.destroyAllWindows() # PERFECT EXIT
                        break
            else:
                break
            
            
           

# Die normalen sequenzen (web) crashen beim Ausstieg - DIESE hier NICHT
#https://stackoverflow.com/questions/54104304/opencv-python-crashes-after-playing-a-video
print("Number of frames in the video: ", frames_counter)
cap.release()
cv2.destroyAllWindows()            