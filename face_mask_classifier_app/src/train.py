import tensorflow as tf
import pandas as pd
from numpy import save
import tqdm
import os
from PIL import Image
import numpy as np
import cv2
from keras_facenet import FaceNet
import pandas as pd 
from keras.utils.np_utils import to_categorical 
from sklearn.preprocessing import LabelBinarizer 

n_classes = 2
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
tf.device('/gpu:0') 
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
tf.compat.v1.Session(config=config)

model=FaceNet()

# Load the train images
df_train = pd.read_csv('/d/project2/Input/train.csv')
filenames_train=df_train[df_train.columns[0]]
labels_train=df_train[df_train.columns[1]]

embed_list_train=[]
label_train_list=[]
k=0
for i in tqdm.tqdm(filenames_train):    
    faces_array_train = model.extract(i, threshold=0.95)
    for j in range(len(faces_array_train)):
        face_embed_dict=faces_array_train[j]
        face_embed=face_embed_dict['embedding']
        embed_list_train.append(face_embed)
        label_train_list.append(labels_train[k])
    k=k+1

embed_list_train=np.array(embed_list_train)
label_train_list=np.array(label_train_list)

lb = LabelBinarizer()
label_train_list = lb.fit_transform(label_train_list)
label_train_list = to_categorical(label_train_list)

df_val = pd.read_csv('/d/project2/Input/val.csv')
filenames_val=df_val[df_val.columns[0]]
labels_val=df_val[df_val.columns[1]]
embed_list_val=[]
label_val_list=[]
k=0
for i in tqdm.tqdm(filenames_val):    
    faces_array_val = model.extract(i, threshold=0.95)
    for j in range(len(faces_array_val)):
        face_embed_dict=faces_array_val[j]
        face_embed=face_embed_dict['embedding']
        embed_list_val.append(face_embed)
        label_val_list.append(labels_val[k])
    k=k+1

embed_list_val=np.array(embed_list_val)
label_val_list=np.array(label_val_list)

lb = LabelBinarizer()
label_val_list = lb.fit_transform(label_val_list)
label_val_list = to_categorical(label_val_list)

np.savez('svm_processed_data.npz', embed_list_train, label_train_list, embed_list_val, label_val_list)

