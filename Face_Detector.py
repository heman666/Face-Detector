import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#read the image
img = cv2.imread('rdj.jpeg')

#convert to gray
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
faces_cordinates = face_cascade.detectMultiScale(gray_img)
#multi scale is used to detect all types of faces like small big all kinds of(diff sizes).
#And detected objects are returned as a list rectangles.
print(faces_cordinates)

#draw rectangles around the face
for (x,y,w,h) in faces_cordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('My Face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Code Completed")
