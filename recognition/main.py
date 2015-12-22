# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:26:29 2015

@author: lochana
"""

import cv2, os
import numpy as np


recognizer = cv2.createLBPHFaceRecognizer()  # Local Binary Patterns Histograms Face Recognizer

def get_images_and_labels(path):
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    images = []
    labels = []
    
    for image_path in image_paths:
        
        image_pil = cv2.imread(image_path,0)
        
        image = np.array(image_pil, 'uint8')
        
        indx = os.path.split(image_path)[1].split(".")[0]
        
        images.append(image)
        labels.append(int(indx))      
        
        cv2.imshow("Adding faces to traning set...", image_pil)
        cv2.waitKey(50)
        
    return images, labels
    
    
path = '/home/lochana/spyder/faceRecog2/images'

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

print "The system has trained successfully."

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

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
    
    elif inpt & 0xFF == ord('r') :
    
        predict_image = np.array(gray, 'uint8')
        
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            
            print "Index {} is Correctly Recognized with confidence {}".format(nbr_predicted, conf)


cap.release()
cv2.destroyAllWindows()


