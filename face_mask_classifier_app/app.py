import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from keras_facenet import FaceNet
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷',
                   layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def mask_image():
    global RGB_img
    # load our serialized face detector model from disk
    #load facenet models
    print("[INFO] loading face detector model...")
    face_detect_model=FaceNet()

    #load mask classifier models
    print("[INFO] loading face mask detector model...")
    #model = load_model(args["model"])
    with open('./face_mask_classifier_app/svm_keral_classifier_final_model.sav', 'rb') as f:
      model = pickle.load(f)

    embed_list_test=[]
    face_box_list=[]
    face_confidence_list=[]

    # load the input image from disk
    image = "./face_mask_classifier_app/images/out.jpg"
    img=cv2.imread(image)
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
    endX =  w
    endY=  h
    #include the probability in the label
    label = "{}: {:.2f}%".format(label, proba[0][0] * 100)
    cv2.putText(img, label, (X, Y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(img, (X, Y), (endX, endY), color, 2)

    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


mask_image()


def mask_detection():
    #local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>',
                unsafe_allow_html=True)
    activities = ["Image"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox(
        "Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>',
                    unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('out.jpg')
            saved_image = st.image(
                image_file, caption='', use_column_width=True)
            st.markdown(
                '<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                st.image(RGB_img, use_column_width=True)

    
mask_detection()
