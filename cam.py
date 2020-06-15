# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:24:19 2020

@author: marti
"""
import cv2

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# cam1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cam1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
# cam2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cam2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while(True):
  ret, frame = cam1.read()
  cv2.imshow('preview',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cam1.release()
cv2.destroyAllWindows() 