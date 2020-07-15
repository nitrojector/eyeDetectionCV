"""
Created on Sat Jun 22 17:27:58 2020

For the purpose of testing new methods

@author: marti
"""

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

cameraNo = 1

cam = cv.VideoCapture(cameraNo)
v, imagesss = cam.read()

saveImgs = False
imgCounter = 0

print('Camera opened')

# Defining reognition profiles
face_cascade = cv.CascadeClassifier('mode/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')
normal_eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')

dt = datetime.now()
t_string = dt.strftime("test%Y%m%d%H%MxCV")

print(t_string)

globalLocIndexX = 0
globalLocIndexY = 0

useCustomThreshold = False

displayEyeDetectionBorder = False

# Whether or not to display visuals[boxes on facial image]
dispVisual = False

dispUselessVisual = False

# Display plot
dispPlot = True

# Display views
dispViews = False

dispEyes = True
    
# Counters for identification
globalCounter = 0
localCounterR = 0
localCounterL = 0
localStartR = 0
localStartL = 0

    
# Threshold for recognizing a look R/L action
lookRightThreshold = 226
lookLeftThreshold = 340


def nothing(x):
    pass

def saveImg(img, tag, index):
    cv.imwrite("./imgs/" + t_string + "/" + tag + str(index) + ".png", img)
    # print("Ran saving method")



cv.namedWindow('Threshold Control')

cv.createTrackbar('[Big]\t\t@L','Threshold Control',lookLeftThreshold,512,nothing)
cv.createTrackbar('[Small]\t\t@R','Threshold Control',lookRightThreshold,512,nothing)

while True:
    ret, frame = cam.read()
    if not ret: 
        cv.waitKey(3)
    
    # Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameCopy = frame
    
    # Recognize faces in frame for eye reognition limiting
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    labelThickness = 1

    kernelSideLen = 10

    # Identification rules
    maxDelayIndex = 6
    minIdentifyL = 4
    minIdentifyR = 4
    
    # Config for eye identification
    eyeFaceRatioMax = 3.2
    eyeFaceRatioMin = 4.5
    eyeDetectionRange = 2/5
    
    # Thresholds for binary processing
    lowAbsThre = 60
    highAbsThre = 255
    
    #The final eye output
    finalEyeLocs = []
    normalEyes = []

    for (x, y, w, h) in faces:
        if dispVisual:
          cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
          cv.putText(frame, 'Face', (x, y + h + 15),
                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), labelThickness)


        locationUpper = int(y+h*eyeDetectionRange)
        
        if displayEyeDetectionBorder and dispVisual:
          cv.rectangle(frame, (x, y), (x + w, locationUpper), (255, 255, 0), 1)
          cv.putText(frame, 'Region', (x, y + int(h*eyeDetectionRange + 15)),
                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), labelThickness)
        
        faceImg = gray[(y):(y + h), x:(x + w)]
        
        maxW = int(((w+h)/2)/eyeFaceRatioMax)
        minW = int(((w+h)/2)/eyeFaceRatioMin)
        
        eyes = eye_cascade.detectMultiScale(faceImg, 1.1, 5, maxSize=(maxW, maxW),
                                            minSize=(minW, minW))
        normalEyes = normal_eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        finalEyeLocs = []
        for (x1, y1, w1, h1) in eyes:
            if y1 + y < locationUpper:
              finalEyeLocs.append((x1, y1, w1, h1))
              # print(y1 + y, locationUpper)
              if dispVisual:
                cv.rectangle(frame,(x1 + x, y1 + y),(x1 + x + w1, y1 + y + h1),
                             (0, 255, 255), 2)
                cv.putText(frame, 'Eye+', (x1 + x, y1 + y + h1 + 15),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                           labelThickness)
        for (x1, y1, w1, h1) in normalEyes:
          if dispVisual and dispUselessVisual:
            cv.rectangle(frame, (x1, y1), (x1+w1,y1+h1), (0, 0, 255), 2)

    
    if len(finalEyeLocs) > 1:
      if not finalEyeLocs[1][0] > finalEyeLocs[0][1]:
        temp = finalEyeLocs[0]
        finalEyeLocs[0] = finalEyeLocs[1]
        finalEyeLocs[1] = temp
      
      eyeImgs = []
      coloredEyeImgs = []
      
      for (x1, y1, w1, h1) in finalEyeLocs:
        eyeImgs.append(gray[(y + y1):(y + y1 + h1), x + x1:(x + x1 + w1)])
        coloredEyeImgs.append(frameCopy[(y+y1):(y+y1+h1),x+x1:(x+x1+w1)])
      
      if saveImgs:
        path = 'L:/OneDrive/Projects/Software/eyeDetectionCV/imgs/' + t_string + "/"
        cv.imwrite(os.path.join(path , 'L' + str(imgCounter) + '.jpg'),
                   coloredEyeImgs[0])
        cv.waitKey(0)
        
        cv.imwrite(os.path.join(path , 'R' + str(imgCounter) + '.jpg'),
                   coloredEyeImgs[1])
        cv.waitKey(0)
      
      # define image processing kernel
      kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                        (kernelSideLen, kernelSideLen))
      
      eyeImgs[0] = cv.resize(eyeImgs[0], (256, 256))
      eyeImgs[1] = cv.resize(eyeImgs[1], (256, 256))
      
      coloredEyeImgs[0] = cv.resize(coloredEyeImgs[0], (256, 256))
      coloredEyeImgs[1] = cv.resize(coloredEyeImgs[1], (256, 256))
      
      # Method 2
      ret1, binaryImg0 = cv.threshold(eyeImgs[0], lowAbsThre ,
                                      highAbsThre, cv.THRESH_BINARY)
      ret2, binaryImg1 = cv.threshold(eyeImgs[1], lowAbsThre,
                                      highAbsThre, cv.THRESH_BINARY)

      closed0 = cv.morphologyEx(binaryImg0, cv.MORPH_CLOSE, kernel)
      closed1 = cv.morphologyEx(binaryImg1, cv.MORPH_CLOSE, kernel)

      gaussKSize = 13
      sobelKSize = 5
      lowThreCanny = 30
      highThreCanny = 40

      cannyOnClosed0 = cv.Canny(closed0, lowThreCanny, highThreCanny)
      cannyOnClosed1 = cv.Canny(closed1, lowThreCanny, highThreCanny)

      gauss0 = cv.GaussianBlur(eyeImgs[0], (gaussKSize, gaussKSize),
                               cv.BORDER_DEFAULT)
      gauss1 = cv.GaussianBlur(eyeImgs[1], (gaussKSize, gaussKSize),
                               cv.BORDER_DEFAULT)

      canny0 = cv.Canny(gauss0, lowThreCanny, highThreCanny)
      canny1 = cv.Canny(gauss1, lowThreCanny, highThreCanny)

      laplacian0 = cv.Laplacian(gauss0, cv.CV_64F)
      laplacian1 = cv.Laplacian(gauss1, cv.CV_64F)

      laplacianOnClosed0 = cv.Laplacian(closed0, cv.CV_64F)
      laplacianOnClosed1 = cv.Laplacian(closed1, cv.CV_64F)

      sobelx0 = cv.Sobel(gauss0, cv.CV_64F, 1, 0, ksize=sobelKSize)  # x
      sobelx1 = cv.Sobel(gauss1, cv.CV_64F, 1, 0, ksize=sobelKSize)  # x

      sobely0 = cv.Sobel(gauss0, cv.CV_64F, 0, 1, ksize=sobelKSize)  # y
      sobely1 = cv.Sobel(gauss1, cv.CV_64F, 0, 1, ksize=sobelKSize)  # y

      # edges = cv.Canny(gray, 20, 30)
      # edges_high_thresh = cv.Canny(gray, 60, 120)
      # cannyDiff = np.hstack((edges, edges_high_thresh))

      # cv.imshow("Canny", cannyDiff)


      contours0, hierarchy0 = cv.findContours(cannyOnClosed0, 1, 2)
      contours1, hierarchy1 = cv.findContours(cannyOnClosed1, 1, 2)
      
      biggestContourIndex0, biggestContourIndex1 = 0, 0
      
      if len(contours0) == 0 or len(contours1) == 0:
        continue
      
      for i in range(1, len(contours0)):
        if cv.contourArea(contours0[i]) > \
                cv.contourArea(contours0[biggestContourIndex0]):
          biggestContourIndex0 = i
          
      for i in range(1, len(contours1)):
        if cv.contourArea(contours1[i]) > \
                cv.contourArea(contours1[biggestContourIndex1]):
          biggestContourIndex1 = i
      
      cnt0 = contours0[biggestContourIndex0]
      cnt1 = contours1[biggestContourIndex1]
      
      M0 = cv.moments(cnt0)
      M1 = cv.moments(cnt1)

      cMx0 = int(M0["m10"] / M0["m00"])
      cMy0 = int(M0["m01"] / M0["m00"])
      cMx1 = int(M1["m10"] / M1["m00"])
      cMy1 = int(M1["m01"] / M1["m00"])
      
      # print(M0)
      
      if len(cnt0) > 5 and len(cnt1) > 5:
        # print('Will draw circles')
        ellipse0 = cv.fitEllipse(cnt0)
        # cv.ellipse(coloredEyeImgs[0], ellipse0, (0, 255, 0), 2)
        
        ellipse1 = cv.fitEllipse(cnt1)
        # cv.ellipse(coloredEyeImgs[1], ellipse1, (0, 255, 0), 2)
      
        (xCNT0,yCNT0),radius0 = cv.minEnclosingCircle(cnt0)
        center0 = (int(xCNT0),int(yCNT0))
        radius0 = int(radius0)
        cv.circle(coloredEyeImgs[0], center0, radius0, (255, 255, 0), 1)
        cv.line(coloredEyeImgs[0], (cMx0, cMy0 + 8),
                (cMx0, cMy0 - 8), (255, 255, 255), 1)
        cv.line(coloredEyeImgs[0], (cMx0 + 8, cMy0),
                (cMx0 - 8, cMy0), (255, 255, 255), 1)
        cv.line(cannyOnClosed0, (cMx0, cMy0 + 8),
                (cMx0, cMy0 - 8), (255, 255, 255), 1)
        cv.line(cannyOnClosed0, (cMx0 + 8, cMy0),
                (cMx0 - 8, cMy0), (255, 255, 255), 1)

        saveImg(coloredEyeImgs[0], "contourR", globalCounter)
        # cv.waitKey(0)

        (xCNT1,yCNT1),radius1 = cv.minEnclosingCircle(cnt1)
        center1 = (int(xCNT1),int(yCNT1))
        radius1 = int(radius1)
        cv.circle(coloredEyeImgs[1], center1, radius1, (255, 255, 0), 1)
        cv.line(coloredEyeImgs[1], (cMx1, cMy1 + 8),
                (cMx1, cMy1 - 8), (255, 255, 255), 1)
        cv.line(coloredEyeImgs[1], (cMx1 + 8, cMy1),
                (cMx1 - 8, cMy1), (255, 255, 255), 1)
        cv.line(cannyOnClosed1, (cMx1, cMy1 + 8),
                (cMx1, cMy1 - 8), (255, 255, 255), 1)
        cv.line(cannyOnClosed1, (cMx1 + 8, cMy1),
                (cMx1 - 8, cMy1), (255, 255, 255), 1)

        saveImg(coloredEyeImgs[1], "contourL", globalCounter)
        # cv.waitKey(0)

        if dispEyes:
            cv.imshow("eyeImg0", coloredEyeImgs[0])
            cv.imshow("eyeImg1", coloredEyeImgs[1])

        eyesR = np.hstack((cannyOnClosed0, canny0, laplacian0,
                           laplacianOnClosed0, sobelx0, sobely0))
        eyesL = np.hstack((cannyOnClosed1, canny1, laplacian1,
                           laplacianOnClosed1, sobelx1, sobely1))

        cv.imshow("[0] Right Eyes", eyesR)
        cv.imshow("[1] Left  Eyes", eyesL)

        globalLocIndexX = xCNT0 + xCNT1
        globalLocIndexY = yCNT0 + yCNT1
        
        if dispPlot:
          plt.plot(xCNT0 + xCNT1, yCNT0 + yCNT1,"g^")
          plt.plot(xCNT0, yCNT0,"ro")
          plt.plot(xCNT1, yCNT1, "bo")
          plt.plot(cMx0 + cMx1, cMy0 + cMy1, "r^")
        
      # Judging which diretion is being looked at
      
      barThreR = cv.getTrackbarPos('[Small]\t@R','Threshold Control')
      barThreL = cv.getTrackbarPos('[BIG]\t@L','Threshold Control')
      
      if globalLocIndexX < (barThreR if useCustomThreshold else lookRightThreshold):
        print("Right Threshold Reached \t@GlobalIndex#", globalCounter,
              "\t@GlobalLocX", globalLocIndexX, "\t< ", lookRightThreshold)
        if localCounterR == 0:
          localStartR = globalCounter
        localCounterR += 1
      
      if globalLocIndexX > (barThreL if useCustomThreshold else lookLeftThreshold):
        print("Left Threshold Reached \t\t@GlobalIndex#", globalCounter,
              "\t@GlobalLocX", globalLocIndexX, "\t> ", lookLeftThreshold)
        if localCounterL == 0:
          localStartL = globalCounter
        localCounterL += 1
        
      if globalCounter - localStartL > maxDelayIndex:
        if localCounterL != 0: print("Left COUNTER CLEARED")
        localCounterL = 0
        
      if globalCounter - localStartR > maxDelayIndex:
        if localCounterR != 0: print("Right COUNTER CLEARED")
        localCounterR = 0
        
      if localCounterR >= minIdentifyR:
        print(">\tR\t>\tR\t>\tR\t>\tR\t>")
        localCounterR = 0
      
      if localCounterL >= minIdentifyL:
        print("<\tL\t<\tL\t<\tL\t<\tL\t<")
        localCounterL = 0
      
      
      # Displaying all the views
      if dispViews:
        # Displaying why the erode process seems different
        # cv.imshow("firstErode", firstErode)
        # cv.imshow("seondDilate", secondDilate)
        # cv.imshow("firstDilate", firstDilate)
        # cv.imshow("secondErode", secondErode)
        
        
        # cv.imshow("opened0", opened0)
        # cv.imshow("opened1", opened1)
        
        # cv.imshow("closed0", closed0)
        # cv.imshow("closed1", closed1)
        
        cv.imshow("canny0", canny0)
        cv.imshow("canny1", canny1)

        
        cv.imshow("binaryImg0", binaryImg0)
        cv.imshow("binaryImg1", binaryImg1)
      
    # Increase loop time stamp counter
    globalCounter += 1
     
    # cv.imshow('Camera', frame)
    frame = cv.resize(frame, (400, 300))
    cv.imshow('Threshold Control',frame)
    
    if dispPlot:
      plt.ylabel("Y Loc")
      plt.xlabel("X Loc")
      plt.axis([0, 512, 0, 512])
      plt.show()
    
    keyValue = cv.waitKey(20)
    if keyValue & 0xff == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
