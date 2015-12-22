# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:33:29 2015

@author: lochana
"""

import numpy as np
import cv2

index=raw_input("Enter the index No. : ")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

cont=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=10,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame',gray)
    inpt=cv2.waitKey(1)
    
    if inpt & 0xFF == ord('q'):
        break
    
    elif inpt & 0xFF == ord('s') :
    
        name='/home/lochana/spyder/faceRecog2/images/'+index+"."+(str(cont))+".png"
        cv2.imwrite(name,gray[y: y + h, x: x + w])
        print cont
        cont+=1

cap.release()
cv2.destroyAllWindows()