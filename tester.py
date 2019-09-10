import cv2
import os
import numpy as np
import faceRecognition as fr
test_img = cv2.imread('C:\\Users\\swara\\Desktop\\Envision\\alia.jpg')
faces_detected,gray_image = fr.faceDetection(test_img)
print("Faces detected : ",faces_detected)
for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(123,255,200),thickness=5)
resized_img = cv2.resize(test_img,(300,500))
cv2.imshow("face detection",test_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()