import cv2 as cv
import numpy as np

img = cv.imread("lady.jpg")
cv.imshow("Lady", img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Lady Gray", gray)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)#play with the 2 hyperparameters for more accuracy, returns a list containing the coordinates of the face found, and len gives the no. of faces found

print(len(face_rect))
for (x,y ,w, h) in face_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=1)
cv.imshow("Detected Face", img)
cv.waitKey(0)