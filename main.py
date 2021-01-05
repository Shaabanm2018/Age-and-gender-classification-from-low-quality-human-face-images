import cv2 as cv
import math
import time
import argparse

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.layers import Input, Dense
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import decode_predictions
from structure import *
import numpy as np
from tkinter import *
from PIL import Image,ImageTk
import os



global frame, frame2
global root, root2

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "Dataset/opencv_face_detector.pbtxt"
faceModel = "Dataset/opencv_face_detector_uint8.pb"

ageProto = "Dataset/age_deploy.prototxt"
ageModel = "Dataset/age_net.caffemodel"

genderProto = "Dataset/gender_deploy.prototxt"
genderModel = "Dataset/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networ
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file orb5 a camera stream
cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20



def load_img():
    global img, image_data, frame
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    global  img, image_data
    frame1 = cv.imread(image_data)
    frameFace, bboxes = getFaceBox(faceNet, frame1)
    for bbox in bboxes:
        face = frame1[max(0,bbox[1]-padding):min(bbox[3]+padding,frame1.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame1.shape[1]-1)]
    
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
    
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))


    tk.Label(frame, text= str("Age").upper() + ': ' + age).pack()
    tk.Label(frame, text= str("Gender").upper() + ': ' + gender).pack()
    
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
             

def enhance_img():
    global image_data, img
    try: 
        temp = img
    except NameError: 
        image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    im = []
    image = pyplot.imread(image_data)
    
    width = int(256)
    hight = int(256)
    dim = (width, hight)
    image = cv2.resize(image,dim, interpolation= cv2.INTER_AREA)
    im.append(image)
    if ((choice == 'low') or (choice == 'blur')):
        auto = autoencoder()
        if (choice == 'low'):
            auto.load_weights(r"Dataset\lowlight3_8m.h5")
        else :
            auto.load_weights("Dataset\blur_8m.h5")
    else :
        auto = pixleated_structre()
        auto.load_weights("Dataset\pix_1m.h5")
    im = np.array(im)

    predictions = auto.predict(im)
    temp = predictions[0].astype('uint8')
    img = temp
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('try.jpg', img)
    display()
    
def display():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()
   

    image_data = r"C:\Users\User\Desktop\evaluation\try.jpg"
    basewidth = 270 # Processing image for dysplaying
    img = Image.open("try.jpg")
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def back():
    global time
    time = 'second'
    cover()
    

def main():
    global frame, img, root 
    root = tk.Toplevel()
    root.title('Portable Image Classifier')
    root.resizable(False, False)
    tit = tk.Label(root, text="Age defult Gender Classification", padx=25, pady=6, font=("", 12)).pack()
    canvas = tk.Canvas(root, height=550, width=550, bg='grey')
    canvas.pack()
    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
        
    chose_image = tk.Button(root, text='Load Image',
                                padx=45, pady=10,
                                fg="white", bg="grey", command=load_img)
    chose_image.pack(side=tk.LEFT)
    
    enhance_image = tk.Button(root, text='Enhance Image',
                                padx=35, pady=10,
                                fg="white", bg="grey", command=enhance_img)
    enhance_image.pack(side=tk.LEFT)
    
    goback = tk.Button(root, text='Back',
                                padx=35, pady=10,
                                fg="white", bg="grey", command=back)
    goback.pack(side=tk.RIGHT)
    
    class_image = tk.Button(root, text='Classify Image', 
                                padx=35, pady=10,
                                fg="white", bg="grey", command=classify)
    class_image.pack(side=tk.LEFT)
    vgg_model = vgg16.VGG16(weights='imagenet')
    root2.destroy()
    root.mainloop()


def low():
    global choice, root
    choice = 'low'
    print(choice)
    main()
    
def blur():
    global choice, root
    choice = 'blur'
    print(choice)
    main()

   
def pixelate():
    global choice, root
    choice = 'pixelate'
    print(choice)
    
    main()
    
def cover():
    global root2, frame2 , time
    root2 = tk.Toplevel()
    root2.title('Portable Image Classifier')
    root2.resizable(False, False)
    tit = tk.Label(root2, text="Age Cover Gender Classification", padx=25, pady=6, font=("", 12)).pack()
    canvas = tk.Canvas(root2, height=500, width=500, bg='grey')
    canvas.pack()
    frame2 = tk.Frame(root2, bg='white')
    for img_display in frame2.winfo_children():
        img_display.destroy()
   
    image_data = r"C:\Users\User\Desktop\evaluation\cover.jpg"
    basewidth = 370 # Processing image for dysplaying
    img = Image.open("cover.jpg")
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame2, text= str('          ').upper()).pack()
    panel_image = tk.Label(frame2, image=img).pack()
    
    
    
    frame2.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
        
    chose_image = tk.Button(root2, text='Low Light Image',
                                padx=45, pady=10,
                                fg="white", bg="grey", command=low)
    chose_image.pack(side=tk.LEFT)
    
    enhance_image = tk.Button(root2, text='Blurred Image',
                                padx=35, pady=10,
                                fg="white", bg="grey", command=blur)
    enhance_image.pack(side=tk.LEFT)
    class_image = tk.Button(root2, text='Pixelated Image', 
                                padx=35, pady=10,
                                fg="white", bg="grey", command=pixelate)
    class_image.pack(side=tk.RIGHT)
    vgg_model = vgg16.VGG16(weights='imagenet')
    if (time == 'second'):
        root.destroy()
    root2.mainloop()
    

    

    
cover()
