"""
Created on Sat Jun 13 13:29:12 2020

@author: marti
"""
import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
cv.namedWindow('v')

# idx, img = cap.read()

# while idx == False:
#   cap = cv.VideoCapture(cameraNo)
#   id, img = cap.read()
#   print("Camera Unavailable")
#   cameraNo += 1
#   if cameraNo == 100:
#     exit(5)

# print("cam available at", cameraNo)

# print('Camera opened')

# cv.namedWindow('eL')
# cv.namedWindow('eR')

# cv.namedWindow('erodeDilate')
# cv.namedWindow('dilateErode')

# cv.namedWindow('closed')
# cv.namedWindow('open')

# Defining reognition profiles
face_cascade = cv.CascadeClassifier('mode/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')
normal_eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')

globalLocIndexX = 0
    
# Counters for identification
# globalCounter = 0
# localCounterR = 0
# localCounterL = 0
# localStartR = 0
# localStartL = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        cv.waitKey(3)
    
    # Grayscale
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameCopy = frame
    
    # Recognize faces in frame for eye reognition limiting
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    
    labelThickness = 1
    
    lookRightThreshold = 260
    lookLeftThreshold = 330
    
    # Identifiation rules
    maxDelayIndex = 10
    minIdentify = 5
    
    # Config
    eyeFaceRatioMax = 3.2
    eyeFaceRatioMin = 4.5
    eyeDetectionRange = 2/5
    displayEyeDetectionBorder = False
    
    lowAbsThre = 60
    highAbsThre = 255
    
    lowAbsThreCanny = 80
    highAbsThreCanny = 255
    
    displayVisual = False
    
    #The final eye output
    finalEyeLocs = []

    for (x, y, w, h) in faces:
        if displayVisual:
          cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
          cv.putText(frame, 'Face', (x, y + h + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), labelThickness)
        
        locationUpper = int(y+h*eyeDetectionRange)
        
        if displayEyeDetectionBorder and displayVisual:
          cv.rectangle(frame, (x, y), (x + w, locationUpper), (255, 255, 0), 2)
          cv.putText(frame, 'Region', (x, y + int(h*eyeDetectionRange + 15)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), labelThickness)
        
        faceImg = grey[(y):(y+h),x:(x+w)]
        
        maxW = int(((w+h)/2)/eyeFaceRatioMax)
        minW = int(((w+h)/2)/eyeFaceRatioMin)
        
        eyes = eye_cascade.detectMultiScale(faceImg, 1.1, 5,maxSize=(maxW,maxW),minSize=(minW,minW))
        normalEyes = normal_eye_cascade.detectMultiScale(grey, 1.3, 5)
        
        finalEyeLocs = []
        for (x1, y1, w1, h1) in eyes:
            if y1 + y < locationUpper:
              finalEyeLocs.append((x1, y1, w1, h1))
              # print(y1 + y, locationUpper)
              if displayVisual:
                cv.rectangle(frame,(x1 + x, y1 + y),(x1 + x + w1, y1 + y + h1), (0, 255, 255), 2)
                cv.putText(frame, 'Eye+', (x1 + x, y1 + y + h1 + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), labelThickness)
        # for (x1, y1, w1, h1) in normalEyes:
        #   if displayVisual:
        #     cv.rectangle(frame, (x1, y1), (x1+w1,y1+h1), (0, 0, 255), 2)
    
    eyeImgs = []
    coloredEyeImgs = []
    
    for (x1, y1, w1, h1) in finalEyeLocs:
      eyeImgs.append(grey[(y+y1):(y+y1+h1),x+x1:(x+x1+w1)])
      coloredEyeImgs.append(frameCopy[(y+y1):(y+y1+h1),x+x1:(x+x1+w1)])
    
    # if len(eyeImgs) > 1:
    #   cv.imshow("eL",eyeImgs[0])
    #   cv.imshow("eR",eyeImgs[1])
    
    if len(finalEyeLocs) > 1:
      kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
      
      eyeImgs[0] = cv.resize(eyeImgs[0], (256, 256))
      eyeImgs[1] = cv.resize(eyeImgs[1], (256, 256))
      
      coloredEyeImgs[0] = cv.resize(coloredEyeImgs[0], (256, 256))
      coloredEyeImgs[1] = cv.resize(coloredEyeImgs[1], (256, 256))
      
      # Method 2
      ret1, binaryImg0 = cv.threshold(eyeImgs[0], lowAbsThre , highAbsThre, cv.THRESH_BINARY)
      ret2, binaryImg1 = cv.threshold(eyeImgs[1], lowAbsThre, highAbsThre, cv.THRESH_BINARY)
      
      
      # Analyzing why the erode process seems different
      # firstErode = cv.erode(binaryImg0,kernel)
      # secondDilate = cv.dilate(firstErode,kernel)
      
      # firstDilate = cv.dilate(binaryImg0,kernel)
      # secondErode = cv.erode(firstDilate,kernel)
      
      
      # Method 2
      # opened0 = cv.morphologyEx(binaryImg0, cv.MORPH_OPEN, kernel)
      # opened1 = cv.morphologyEx(binaryImg1, cv.MORPH_OPEN, kernel)
      
      # Method 1
      # opened0 = cv.morphologyEx(eyeImgs[0], cv.MORPH_OPEN, kernel)
      # opened1 = cv.morphologyEx(eyeImgs[1], cv.MORPH_OPEN, kernel)
      
      # Method 2
      # Because the black values are likely 0 and the white ones 1, 
      # Opened and Closed seems to work as if excahnged for the two 
      # processes
      closed0 = cv.morphologyEx(binaryImg0, cv.MORPH_CLOSE, kernel)
      closed1 = cv.morphologyEx(binaryImg1, cv.MORPH_CLOSE, kernel)
      
      # Drawing borders
      bordered0 = cv.Canny(closed0, lowAbsThreCanny, highAbsThreCanny)
      bordered1 = cv.Canny(closed1, lowAbsThreCanny, highAbsThreCanny)
      
      # Method 1
      # ret1, binaryImg0 = cv.threshold(opened0, lowAbsThre , highAbsThre, cv.THRESH_BINARY)
      # ret2, binaryImg1 = cv.threshold(opened1, lowAbsThre, highAbsThre, cv.THRESH_BINARY)
      
      # cnt0, hierarchy0 = cv.findContours(closed0, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      # cnt1, hierarchy1 = cv.findContours(closed1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      
      contours0, hierarchy0 = cv.findContours(bordered0, 1, 2)
      contours1, hierarchy1 = cv.findContours(bordered1, 1, 2)
      
      biggestContourIndex0, biggestContourIndex1 = 0, 0
      
      # print(len(contours0), len(contours1))
      # cnt0 = contours0[biggestContourIndex0]
      # cnt1 = contours1[biggestContourIndex1]
      
      if len(contours0) == 0 or len(contours1) == 0:
        continue
      
      for i in range(1, len(contours0)):
        if cv.contourArea(contours0[i]) > cv.contourArea(contours0[biggestContourIndex0]):
          biggestContourIndex0 = i
          
      for i in range(1, len(contours1)):
        if cv.contourArea(contours1[i]) > cv.contourArea(contours1[biggestContourIndex1]):
          biggestContourIndex1 = i
      
      # cnt0 = max(contours0, key=cv.contourArea(contours0))
      # cnt1 = max(contours1, key=cv.contourArea(contours1))
      
      # for i in range(len(contours0)):
      #   cv.drawContours(coloredEyeImgs[0], contours0[i], i, (255,255,0), 1)
        
      # for i in range(len(contours1)):
      #   cv.drawContours(coloredEyeImgs[1], contours1[i], i, (255,255,0), 1)
      
      cnt0 = contours0[biggestContourIndex0]
      cnt1 = contours1[biggestContourIndex1]
      
      M0 = cv.moments(cnt0)
      M1 = cv.moments(cnt1)
      
      # print(M0,M1)
      
      if len(cnt0) > 5 and len(cnt1) > 5:
        # print('Will draw circles')
        ellipse0 = cv.fitEllipse(cnt0)
        # cv.ellipse(coloredEyeImgs[0], ellipse0, (0, 255, 0), 2)
        
        ellipse1 = cv.fitEllipse(cnt1)
        # cv.ellipse(coloredEyeImgs[1], ellipse1, (0, 255, 0), 2)
      
        (xCNT0,yCNT0),radius0 = cv.minEnclosingCircle(cnt0)
        center0 = (int(xCNT0),int(yCNT0))
        radius0 = int(radius0)
        cv.circle(coloredEyeImgs[0], center0, radius0, (255, 255, 0), 2)
        
        (xCNT1,yCNT1),radius1 = cv.minEnclosingCircle(cnt1)
        center1 = (int(xCNT1),int(yCNT1))
        radius1 = int(radius1)
        cv.circle(coloredEyeImgs[1], center1, radius1, (255, 255, 0), 2)
        
        # globalLocIndexX = xCNT0 + xCNT1
        
        # plt.plot(xCNT0 + xCNT1, yCNT0 + yCNT1,"g^")
        # plt.plot(xCNT0, yCNT0,"ro")
        # plt.plot(xCNT1, yCNT1, "bo")
        
      # Judging which diretion is being looked at
      #
        # if localCounterL == 0:
        #   localStartL = globalCounter
        # localCounterL += 1
      
      # if globalLocIndexX < lookRightThreshold:
      #   print("Right Threshold Reached @GlobalIndex#",  "\t\t@GlobalLoc", globalLocIndexX)
        # if localCounterR == 0:
        #   localStartR = globalCounter
        # localCounterR += 1
        
      # if globalCounter - localStartL > maxDelayIndex:
      #   localCounterL = 0
        
      # if globalCounter - localStartR > maxDelayIndex:
      #   localCounterR = 0
        
      # if localCounterR >= minIdentify:
      #   print("\t\t\tYou Looked Right!")
      #   localCounterR = 0
      
      # if localCounterL >= minIdentify:
      #   print("\t\t\tYou Looked Left!")
      #   localCounterL = 0
      
      
      
      
      
      # Displaying why the erode process seems different
      # cv.imshow("firstErode", firstErode)
      # cv.imshow("seondDilate", secondDilate)
      # cv.imshow("firstDilate", firstDilate)
      # cv.imshow("secondErode", secondErode)
      
      
      # cv.imshow("opened0", opened0)
      # cv.imshow("opened1", opened1)
      
      # cv.imshow("closed0", closed0)
      # cv.imshow("closed1", closed1)
      
      cv.imshow("bordered0", bordered0)
      cv.imshow("bordered1", bordered1)
      
      # cv.imshow("binaryImg0", binaryImg0)
      # cv.imshow("binaryImg1", binaryImg1)
      
      cv.imshow("eyeImg0", coloredEyeImgs[0])
      cv.imshow("eyeImg1", coloredEyeImgs[1])
      
    # globalCounter += 1
    
    cv.imshow('v', frame)
    # plt.ylabel("Y Loc")
    # plt.xlabel("X Loc")
    # plt.axis([0, 512, 0, 512])
    # plt.show()
    
    keyvalue = cv.waitKey(20)
    if keyvalue & 0xff == ord('q'):
        break

cv.destroyAllWindows()
