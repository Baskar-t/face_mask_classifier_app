
# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
from keras_facenet import FaceNet
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# construct the argument parser and parse the argumen#ts
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
args = vars(ap.parse_args())


#load facenet models
print("[INFO] loading face detector model...")
face_detect_model=FaceNet()

#load mask classifier models
print("[INFO] loading face mask detector model...")
#model = load_model(args["model"])
model = pickle.load(open(args["model"], 'rb'))

embed_list_test=[]
face_box_list=[]
face_confidence_list=[]

# load the input image from disk
image = args["image"]
img=cv2.imread(args["image"])
faces_array_train = face_detect_model.extract(image, threshold=0.95)
for j in range(len(faces_array_train)):
    face_embed_dict=faces_array_train[j]
    face_embed=face_embed_dict['embedding']
    face_confidence=face_embed_dict['confidence']
    face_box=face_embed_dict['box']
    embed_list_test.append(face_embed)
    face_box_list.append(face_box)
    face_confidence_list.append(face_confidence)

face=np.array(embed_list_test)
#for i in range(len(face)):
label_pred = model.predict(face)
proba = model.predict_proba(face)
if label_pred == 0:
    label = "Mask"
else:
    label= "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
X=face_box_list[0][0]
Y=face_box_list[0][1]
w=face_box_list[0][2]
h=face_box_list[0][3]
(X, Y) = max(0, X), max(0, Y)
endX = min(w - 1, w)
endY= min(h - 1, h)
#include the probability in the label
label = "{}: {:.2f}%".format(label, proba[0][0] * 100)
cv2.putText(img, label, (X, Y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
cv2.rectangle(img, (X, Y), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)