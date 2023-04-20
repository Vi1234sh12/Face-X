# Face filters (Snapchat like) using OpenCV
# @author:PSY222

import cv2
import sys
import datetime as dt
from time import sleep
import numpy as np
import os

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, 'img/minion_eyes.png')
cascPath = os.path.join(current_dir, 'frontalEyes.xml')  # for eye detection

if os.path.exists(cascPath):
    print('Loading camera now')
else:
    print('CascPath does not exists')

eyeCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
anterior = 0
eyes = cv2.imread(image_path)


def put_eyes(eyes,fc,x,y,w,h):
    eye_width,eye_height = w, h
    eye = cv2.resize(eyes,(w,h))

    for i in range(eye.shape[0]):
         for j in range(eye.shape[1]):
            for k in range(3):
                if not np.all(eye[i,j,k]==[0,0,0]):
                    fc[y+i,x+j][k] = eye[i,j][k]
    return fc
    
    
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_loc = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    for (x, y, w, h) in eye_loc:
            frame = put_eyes(eyes,frame,x,y,w,h)
            

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()