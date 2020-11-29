"""
Created on Sat Jun 22 17:27:58 2020

@author: marti

For the purpose of testing new methods
"""

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import simple_cb as cba

cameraNo = 0

cam = cv.VideoCapture(cameraNo)
v, imagesss = cam.read()

saveImgs = False
saveFaces = False
imgCounter = 0
faceCounter = 0

print('Camera opened')

# Defining reognition profiles
face_cascade = cv.CascadeClassifier('mode/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')
normal_eye_cascade = cv.CascadeClassifier('mode/haarcascade_eye.xml')

dt = datetime.now()
t_string = dt.strftime("t%Y%m%d%H%M%S_")
folder = input("Please enter session id > ")
newpath = './imgs/' + folder
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(t_string)

globalLocIndexX = 0
globalLocIndexY = 0

displayEyeDetectionBorder = False

# Whether or not to display visuals[boxes on facial image]
dispVisual = False

dispUselessVisual = False

imgProcessMode = False

# Display plot
dispPlot = False

# Display views
dispViews = False

dispEyes = True

imgNo = 1


# Counters for identification
globalCounter = 0
localCounterR = 0
localCounterL = 0
localStartR = 0
localStartL = 0

# Threshold for recognizing a look R/L action
lookRightThreshold = 226
lookLeftThreshold = 315

labelThickness = 1

kernelSideLen = 10

cbaPercent = 50

# Identification rules
maxDelayIndex = 6
minIdentifyL = 4
minIdentifyR = 4

# Config for eye identification
eyeFaceRatioMax = 3.2
eyeFaceRatioMin = 4.5
eyeDetectionRange = 2 / 5

# Thresholds for binary processing
lowAbsThre = 65
highAbsThre = 255

# Edge Thresh
gaussKSize = 17
sobelKSize = 5
lowThreCanny = 40
highThreCanny = 70

# Experiment Testing
eyesFrameErrors = 0
eyesLocalErrors = 0
eyesLocalWithScaleErrors = 0


def nothing(x):
    pass


def Sobel(image):
    grad_x = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    grad_y = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    return cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def Prewitt(image):
    prewittkx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    img_prewittx = cv.filter2D(image, -1, prewittkx)
    img_prewitty = cv.filter2D(image, -1, prewittky)

    return img_prewittx + img_prewitty



def saveImg(img, tag, index):
    path = './imgs/' + folder + '/' + str(index) + tag + '.png'
    cv.imwrite(path, img)
    # print("Ran saving method")


cv.namedWindow('Threshold Control')

cv.createTrackbar('>L', 'Threshold Control', lookLeftThreshold, 512, nothing)
cv.createTrackbar('<R', 'Threshold Control', lookRightThreshold, 512, nothing)
cv.createTrackbar('cannyLo', 'Threshold Control', lowThreCanny, 150, nothing)
cv.createTrackbar('cannyHi', 'Threshold Control', highThreCanny, 150, nothing)
cv.createTrackbar('binLo', 'Threshold Control', lowAbsThre, 255, nothing)
cv.createTrackbar('binHi', 'Threshold Control', highAbsThre, 255, nothing)
cv.createTrackbar('closedK', 'Threshold Control', kernelSideLen, 64, nothing)
cv.createTrackbar('cbaP', 'Threshold Control', cbaPercent, 100, nothing)


while True:

    ret, frame = cam.read()
    if not ret:
        cv.waitKey(3)

    cbaPercent = cv.getTrackbarPos('cbaP', 'Threshold Control')

    # Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameCopy = frame

    # Recognize faces in frame for eye reognition limiting
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # The final eye output
    finalEyeLocs = []
    eyesLocal = []
    normalEyes = []

    for (x, y, w, h) in faces:
        if dispVisual:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv.putText(frame, 'Face', (x, y + h + 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), labelThickness)

        locationUpper = int(y + h * eyeDetectionRange)

        if displayEyeDetectionBorder and dispVisual:
            cv.rectangle(frame, (x, y), (x + w, locationUpper), (255, 255, 0), 1)
            cv.putText(frame, 'Region', (x, y + int(h * eyeDetectionRange + 15)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), labelThickness)

        faceImg = gray[y:(y + h), x:(x + w)]

        maxW = int(((w + h) / 2) / eyeFaceRatioMax)
        minW = int(((w + h) / 2) / eyeFaceRatioMin)

        eyesLocal = eye_cascade.detectMultiScale(faceImg, 1.1, 5)

        eyes = eye_cascade.detectMultiScale(faceImg, 1.1, 5, maxSize=(maxW, maxW),
                                            minSize=(minW, minW))
        normalEyes = normal_eye_cascade.detectMultiScale(gray, 1.3, 5)

        finalEyeLocs = []
        for (x1, y1, w1, h1) in eyes:
            if y1 + y < locationUpper:
                finalEyeLocs.append((x1, y1, w1, h1))
                # print(y1 + y, locationUpper)
                if dispVisual:
                    cv.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w1, y1 + y + h1),
                                 (0, 255, 255), 2)
                    cv.putText(frame, 'Eye', (x1 + x, y1 + y + h1 + 15),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                               labelThickness)
        for (x1, y1, w1, h1) in normalEyes:
            if dispVisual and dispUselessVisual:
                cv.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

    if len(normalEyes) != 2:
        eyesFrameErrors += 1
    if len(eyesLocal) != 2:
        eyesLocalErrors += 1
    if len(finalEyeLocs) != 2:
        eyesLocalWithScaleErrors += 1

    if globalCounter >= 600:
        print("Face Localization Technique Accuracies")
        print("Whole Frame Identification:\t\t" + str(100 - eyesFrameErrors/globalCounter*100) + "%")
        print("Local Face Identification:\t\t" + str(100 - eyesLocalErrors/globalCounter*100) + "%")
        print("Local with Scale Identi. :\t\t" + str(100 - eyesLocalWithScaleErrors/globalCounter*100) + "%")
        exit(402)


    if len(finalEyeLocs) > 1:
        if not finalEyeLocs[1][0] > finalEyeLocs[0][1]:
            temp = finalEyeLocs[0]
            finalEyeLocs[0] = finalEyeLocs[1]
            finalEyeLocs[1] = temp

        eyeImgs = []
        coloredEyeImgs = []

        for (x1, y1, w1, h1) in finalEyeLocs:
            eyeImgs.append(gray[(y + y1):(y + y1 + h1), x + x1:(x + x1 + w1)])
            coloredEyeImgs.append(frameCopy[(y + y1):(y + y1 + h1), x + x1:(x + x1 + w1)])

        kernelSideLen =cv.getTrackbarPos('closedK', 'Threshold Control')

        # define image processing kernel
        kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                          (kernelSideLen, kernelSideLen))

        eyeImgs[0] = cv.resize(eyeImgs[0], (256, 256))
        eyeImgs[1] = cv.resize(eyeImgs[1], (256, 256))

        if saveImgs:
            saveImg(coloredEyeImgs[0], "R", imgCounter)
            cv.waitKey(20)
            saveImg(coloredEyeImgs[1], "L", imgCounter)
            imgCounter += 1


        coloredEyeImgs[0] = cv.resize(coloredEyeImgs[0], (256, 256))
        coloredEyeImgs[1] = cv.resize(coloredEyeImgs[1], (256, 256))

        lowThreCanny = cv.getTrackbarPos('cannyLo', 'Threshold Control')
        highThreCanny = cv.getTrackbarPos('cannyHi', 'Threshold Control')

        lowAbsThre = cv.getTrackbarPos('binLo', 'Threshold Control')
        highAbsThre = cv.getTrackbarPos('binHi', 'Threshold Control')

        gauss0 = cv.GaussianBlur(eyeImgs[0], (gaussKSize, gaussKSize),
                                 cv.BORDER_DEFAULT)
        gauss1 = cv.GaussianBlur(eyeImgs[1], (gaussKSize, gaussKSize),
                                 cv.BORDER_DEFAULT)

        ret1, binaryImg0 = cv.threshold(eyeImgs[0], lowAbsThre,
                                        highAbsThre, cv.THRESH_BINARY)
        ret2, binaryImg1 = cv.threshold(eyeImgs[1], lowAbsThre,
                                        highAbsThre, cv.THRESH_BINARY)

        closed0 = cv.morphologyEx(binaryImg0, cv.MORPH_CLOSE, kernel)
        closed1 = cv.morphologyEx(binaryImg1, cv.MORPH_CLOSE, kernel)

        # opened0 = cv.morphologyEx(binaryImg0, cv.MORPH_OPEN, kernel)
        # opened1 = cv.morphologyEx(binaryImg1, cv.MORPH_OPEN, kernel)

        cannyOnClosed0 = cv.Canny(closed0, 30, 40)
        cannyOnClosed1 = cv.Canny(closed1, 30, 40)

        canny0 = cv.Canny(eyeImgs[0], lowThreCanny, highThreCanny)
        canny1 = cv.Canny(eyeImgs[1], lowThreCanny, highThreCanny)

        # laplacian0 = cv.Laplacian(gauss0, cv.CV_64F)
        # laplacian1 = cv.Laplacian(gauss1, cv.CV_64F)

        laplacian0 = cv.Laplacian(closed0, cv.CV_64F)
        laplacian1 = cv.Laplacian(closed1, cv.CV_64F)

        prewitt0 = Prewitt(closed0)
        prewitt1 = Prewitt(closed1)

        sobel0 = Sobel(closed0)
        sobel1 = Sobel(closed1)

        # edges = cv.Canny(gray, 20, 30)
        # edges_high_thresh = cv.Canny(gray, 60, 120)
        # cannyDiff = np.hstack((edges, edges_high_thresh))

        # cv.imshow("Canny", cannyDiff)

        contours0, hierarchy0 = cv.findContours(cannyOnClosed0, 1, 2)
        contours1, hierarchy1 = cv.findContours(cannyOnClosed1, 1, 2)

        contours2, hierarchy2 = cv.findContours(canny0, 1, 2)
        contours3, hierarchy3 = cv.findContours(canny1, 1, 2)

        biggestContourIndex0, biggestContourIndex1, biggestContourIndex2, biggestContourIndex3 = 0, 0, 0, 0

        if len(contours0) == 0 or len(contours1) == 0 or \
                len(contours2) == 0 or len(contours3) == 0:
            continue

        for i in range(1, len(contours0)):
            if cv.contourArea(contours0[i]) > \
                    cv.contourArea(contours0[biggestContourIndex0]):
                biggestContourIndex0 = i

        for i in range(1, len(contours1)):
            if cv.contourArea(contours1[i]) > \
                    cv.contourArea(contours1[biggestContourIndex1]):
                biggestContourIndex1 = i

        for i in range(1, len(contours2)):
            if cv.contourArea(contours2[i]) > \
                    cv.contourArea(contours2[biggestContourIndex2]):
                biggestContourIndex2 = i

        for i in range(1, len(contours3)):
            if cv.contourArea(contours3[i]) > \
                    cv.contourArea(contours3[biggestContourIndex3]):
                biggestContourIndex3 = i

        # cv.drawContours(coloredEyeImgs[0], contours2, -1, (0, 0, 255), 2)
        # cv.drawContours(coloredEyeImgs[1], contours3, -1, (0, 0, 255), 2)
        #
        # cv.drawContours(coloredEyeImgs[0], contours2, biggestContourIndex2, (0, 255, 255), 2)
        # cv.drawContours(coloredEyeImgs[1], contours3, biggestContourIndex3, (0, 255, 255), 2)

        cnt0 = contours0[biggestContourIndex0]
        cnt1 = contours1[biggestContourIndex1]
        cnt2 = contours2[biggestContourIndex2]
        cnt3 = contours3[biggestContourIndex3]

        M0 = cv.moments(cnt0)
        M1 = cv.moments(cnt1)
        M2 = cv.moments(cnt2)
        M3 = cv.moments(cnt3)

        try:
            cMx0 = int(M0["m10"] / M0["m00"])
            cMy0 = int(M0["m01"] / M0["m00"])
            cMx1 = int(M1["m10"] / M1["m00"])
            cMy1 = int(M1["m01"] / M1["m00"])
            # cMx2 = int(M2["m10"] / M2["m00"])
            # cMy2 = int(M2["m01"] / M2["m00"])
            # cMx3 = int(M3["m10"] / M3["m00"])
            # cMy3 = int(M3["m01"] / M3["m00"])
        except ZeroDivisionError:
            continue

        # print(M0)

        if len(cnt0) > 5 and len(cnt1) > 5:
            # print('Will draw circles')
            # ellipse2 = cv.fitEllipse(cnt2)
            # cv.ellipse(canny0, ellipse2, (255, 255, 255), 2)
            #
            # ellipse3 = cv.fitEllipse(cnt3)
            # cv.ellipse(canny1, ellipse3, (255, 255, 255), 2)

            (xCNT2, yCNT2), radius2 = cv.minEnclosingCircle(cnt2)
            (xCNT3, yCNT3), radius3 = cv.minEnclosingCircle(cnt3)

            center2 = (int(xCNT2), int(yCNT2))
            radius2 = int(radius2)
            center3 = (int(xCNT3), int(yCNT3))
            radius3 = int(radius3)

            cv.circle(canny0, center2, radius2, (255, 255, 255), 1)
            cv.circle(canny1, center3, radius3, (255, 255, 255), 1)

            (xCNT0, yCNT0), radius0 = cv.minEnclosingCircle(cnt0)
            xCNT0 = int(xCNT0)
            yCNT0 = int(yCNT0)
            center0 = (xCNT0, yCNT0)
            radius0 = int(radius0)
            # cv.circle(coloredEyeImgs[0], center0, radius0, (255, 255, 0), 1)
            cv.line(coloredEyeImgs[0], (xCNT0, yCNT0 + 8),
                    (xCNT0, yCNT0 - 8), (255, 255, 0), 1)
            cv.line(coloredEyeImgs[0], (xCNT0 + 8, yCNT0),
                    (xCNT0 - 8, yCNT0), (255, 255, 0), 1)
            cv.line(coloredEyeImgs[0], (cMx0, cMy0 + 8),
                    (cMx0, cMy0 - 8), (255, 255, 255), 1)
            cv.line(coloredEyeImgs[0], (cMx0 + 8, cMy0),
                    (cMx0 - 8, cMy0), (255, 255, 255), 1)
            cv.line(cannyOnClosed0, (cMx0, cMy0 + 8),
                    (cMx0, cMy0 - 8), (255, 255, 255), 1)
            cv.line(cannyOnClosed0, (cMx0 + 8, cMy0),
                    (cMx0 - 8, cMy0), (255, 255, 255), 1)
            # cv.line(canny0, (cMx2, cMy2 + 8),
            #         (cMx2, cMy2 - 8), (255, 255, 255), 1)
            # cv.line(canny0, (cMx2 + 8, cMy2),
            #         (cMx2 - 8, cMy2), (255, 255, 255), 1)

            (xCNT1, yCNT1), radius1 = cv.minEnclosingCircle(cnt1)
            xCNT1 = int(xCNT1)
            yCNT1 = int(yCNT1)
            center1 = (xCNT1, yCNT1)
            radius1 = int(radius1)
            # cv.circle(coloredEyeImgs[1], center1, radius1, (255, 255, 0), 1)
            cv.line(coloredEyeImgs[1], (xCNT1, yCNT1 + 8),
                    (xCNT1, yCNT1 - 8), (255, 255, 0), 1)
            cv.line(coloredEyeImgs[1], (xCNT1 + 8, yCNT1),
                    (xCNT1 - 8, yCNT1), (255, 255, 0), 1)
            cv.line(coloredEyeImgs[1], (cMx1, cMy1 + 8),
                    (cMx1, cMy1 - 8), (255, 255, 255), 1)
            cv.line(coloredEyeImgs[1], (cMx1 + 8, cMy1),
                    (cMx1 - 8, cMy1), (255, 255, 255), 1)
            cv.line(cannyOnClosed1, (cMx1, cMy1 + 8),
                    (cMx1, cMy1 - 8), (255, 255, 255), 1)
            cv.line(cannyOnClosed1, (cMx1 + 8, cMy1),
                    (cMx1 - 8, cMy1), (255, 255, 255), 1)
            # cv.line(canny1, (cMx3, cMy3 + 8),
            #         (cMx3, cMy3 - 8), (255, 255, 255), 1)
            # cv.line(canny1, (cMx3 + 8, cMy3),
            #         (cMx3 - 8, cMy3), (255, 255, 255), 1)

            colorEyes = np.vstack((coloredEyeImgs[1],coloredEyeImgs[0]))
            colorEyes = cv.resize(colorEyes, (384, 768))

            if dispEyes:
                cv.imshow("colorEyes", colorEyes)

            eyesR = np.hstack((closed0, cannyOnClosed0, sobel0, laplacian0, prewitt0))
            eyesL = np.hstack((closed1, cannyOnClosed1, sobel1, laplacian1, prewitt1))

            # eyesR = np.hstack((cannyOnClosed0, closed0, opened0, canny0))
            # eyesL = np.hstack((cannyOnClosed1, closed1, opened1, canny1))

            # eyesR = np.hstack((binaryImg0, binaryGImg0, closed0, closedG0))
            # eyesL = np.hstack((binaryImg1, binaryGImg1, closed1, closedG1))
            allEyes = np.vstack((eyesL, eyesR))

            cv.imshow("Eyes", allEyes)

            globalLocIndexX = int(xCNT0 + xCNT1)
            globalLocIndexY = int(yCNT0 + yCNT1)

            if dispPlot:
                plt.plot(xCNT0 + xCNT1, yCNT0 + yCNT1, "g^")
                plt.plot(xCNT0, yCNT0, "ro")
                plt.plot(xCNT1, yCNT1, "bo")
                plt.plot(cMx0 + cMx1, cMy0 + cMy1, "r^")

        # Judging which diretion is being looked at

        lookRightThreshold = cv.getTrackbarPos('<R', 'Threshold Control')
        lookLeftThreshold = cv.getTrackbarPos('>L', 'Threshold Control')
        # print("L", barThreL, "R", barThreR)

        if globalLocIndexX < lookRightThreshold:
            print("Right Threshold Reached \t@GlobalIndex#", globalCounter,
                  "\t@GlobalLocX", globalLocIndexX, " < ", lookRightThreshold)
            if localCounterR == 0:
                localStartR = globalCounter
            localCounterR += 1

        if globalLocIndexX > lookLeftThreshold:
            print("Left Threshold Reached \t\t@GlobalIndex#", globalCounter,
                  "\t@GlobalLocX", globalLocIndexX, " > ", lookLeftThreshold)
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
    cv.imshow('Threshold Control', frame)
    if saveFaces:
        saveImg(frame, "F", faceCounter)
        faceCounter += 1

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
