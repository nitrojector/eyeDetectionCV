"""
Created on 14:39 Nov 29, 2020

A class with different methods for
finding patterns of an optimal threshold
in binary processing



@On start
request folder name
    folder name for current session to be inspected
    Path = "./imgs/%folder_name%/"



@While running
:key "r"
    confirms the threshold of the current image and
    continues to the next

:key "q"
    quits the program and saves dataset for inspected
    images

:key "s"
    skips current image for inspection
    possibility of bad image, etc.


Extra polar adjustments:

:key "," "."
    adjusts the threshold for binary processing (- / +)

:key ";" "'"
    adjusts the

:key "l" "k"
    adjusts the intensity for polarizing pixels of current
    image (- / +)

"""



from time import time
import cv2 as cv
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


def nothing(x):
    return


def shrink(data, scale):
    output = np.array([])
    for ind in range(0, len(data), scale):
        output = np.append(output, [sum(data[ind:ind + scale])])
    return output


def getOddElements(list):
    oddElements = []
    for id, element in enumerate(list):
        if not id % 2 == 0:
            oddElements.append(element)
    return oddElements


def getCenter(img):
    global showContours
    img = cv.Canny(img, 30, 40)
    contours = cv.findContours(img, 1, 2)[0]
    contours = getOddElements(contours)

    if len(contours) == 0:
        print('!!!contour error!!!')
        return 128, 128, 10
    maxContInd = 0
    # print(str(len(contours)) + ' contours scanned.')
    for i in range(1, len(contours)):
        if cv.contourArea(contours[i]) > \
                cv.contourArea(contours[maxContInd]):
            maxContInd = i
    (locX, locY), rCircle = cv.minEnclosingCircle(contours[maxContInd])
    if showContours:
        mask = np.zeros(img.shape, np.uint8)
        cv.drawContours(mask, contours, -1, (255), 1)
        cv.rectangle(mask, (0, 0), (255, 255), (255))
        allCont = copy.deepcopy(mask)
        for idx, contour in enumerate(contours):
            mask = np.zeros(img.shape, np.uint8)
            cv.rectangle(mask, (0, 0), (255, 255), (255))
            cv.drawContours(mask, [contour], -1, (255), 1)
            allCont = np.hstack((allCont, mask))
        cv.imshow('Contours', allCont)
    return locX, locY, rCircle


def polarize(img, intensity, center):
    global consderDimensions
    if (not img.shape == (256,256) or not considerDimensions) and (center < 255 and center > 0):
        print('Invalid Parameters or Dimensions, Original Image Returned')
        return img
    polarImg = np.zeros(shape=(256, 256), dtype=np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pxVal = img[x][y]
            if pxVal < center:
                polarImg[x][y] = np.math.ceil(pxVal/intensity)
            else:
                tempVal = 255 - np.math.ceil((255 - pxVal) / intensity)
                polarImg[x][y] = tempVal
    return polarImg


considerDimensions = True

polarIntensity = 1
polarCenter = 128

threInfo128 = np.ndarray(shape=(0, 131))
threInfo64 = np.ndarray(shape=(0, 67))
threInfo32 = np.ndarray(shape=(0, 35))

loThreVal = 65
hiThreVal = 255

showContours = True


folderName = input('Folder Name > ')
fileList = os.listdir('./imgs/' + folderName)

try:
    os.mkdir('./dat/' + folderName)
except OSError:
    print('Folder already exist')
counter = 1

cv.namedWindow('Folder ' + folderName)


for fileName in fileList:
    path = './imgs/' + folderName + '/' + fileName
    img = cv.cvtColor(cv.resize(cv.imread(path), (256, 256)), cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (17, 17), cv.BORDER_DEFAULT)

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist128 = shrink(hist, 2)
    hist64 = shrink(hist, 4)
    hist32 = shrink(hist, 8)

    plt.hist(img.ravel(), 256, [0, 256])

    while True:
        startTime = time()
        skip = False
        ret, binImg = cv.threshold(img, loThreVal, hiThreVal, cv.THRESH_BINARY)
        binImg = cv.morphologyEx(binImg, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (10, 10)))
        # polarImg = polarize(img, polarIntensity, polarCenter)
        dispImg = np.vstack((img, binImg)) #, polarImg
        # cv.putText(dispImg, 'C: ' + str(polarCenter) + ' | I: ' + str(round(polarIntensity, 2)), (5, 532),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        X, Y, R = getCenter(binImg)
        cv.line(dispImg, (int(X), int(Y + R)), (int(X), int(Y - R)), (255, 255, 255), 1)
        cv.line(dispImg, (int(X + R), int(Y)), (int(X - R), int(Y)), (255, 255, 255), 1)

        cv.putText(dispImg, str(loThreVal) + ' | ' + str(hiThreVal) + '  ' + str(counter) + ' out of ' +
                   str(len(fileList)), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv.putText(dispImg, fileName, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv.imshow('Folder ' + folderName, dispImg)

        # print('The last epoch took ' + str(int((time() - startTime)*1000)) + 'ms !')

        keyVal = cv.waitKey(20)
        if keyVal & 0xff == ord('r'):
            break
        elif keyVal & 0xff == ord('s'):
            skip = True
            break
        elif keyVal & 0xff == ord('l'):
            polarIntensity += 0.05
        elif keyVal & 0xff == ord('k'):
            polarIntensity -= 0.05
        elif keyVal & 0xff == ord('\''):
            polarCenter += 2
        elif keyVal & 0xff == ord(';'):
            polarCenter -= 2
        elif keyVal & 0xff == ord('.'):
            loThreVal += 2
        elif keyVal & 0xff == ord(','):
            loThreVal -= 2
            if loThreVal <= 10:
                loThreVal += 2
                print('!!! Threshold TOO Small !!! [Change reverted]')
        elif keyVal & 0xff == ord('q'):
            np.savetxt('./dat/' + folderName + '/data128.csv', threInfo128, delimiter=',', fmt='%s')
            np.savetxt('./dat/' + folderName + '/data64.csv', threInfo64, delimiter=',', fmt='%s')
            np.savetxt('./dat/' + folderName + '/data32.csv', threInfo32, delimiter=',', fmt='%s')
            exit(202)
        elif keyVal & 0xff == ord('='):
            exit(203)

    counter += 1

    print(np.append([fileName, loThreVal, hiThreVal], hist))
    if not skip:
        threInfo128 = np.append(threInfo128, [np.append([fileName, loThreVal, hiThreVal], hist128)], axis=0)
        threInfo64 = np.append(threInfo64, [np.append([fileName, loThreVal, hiThreVal], hist64)], axis=0)
        threInfo32 = np.append(threInfo32, [np.append([fileName, loThreVal, hiThreVal], hist32)], axis=0)
        plt.savefig('./dat/' + folderName + '/' + fileName)
    plt.show()

np.savetxt('./dat/' + folderName + '/data128.csv', threInfo128, delimiter=',', fmt='%s')
np.savetxt('./dat/' + folderName + '/data64.csv', threInfo64, delimiter=',', fmt='%s')
np.savetxt('./dat/' + folderName + '/data32.csv', threInfo32, delimiter=',', fmt='%s')
    
exit(200)