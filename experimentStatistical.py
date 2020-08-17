import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

lookRightThreshold = 226
lookLeftThreshold = 315
lowAbsThre = 65
highAbsThre = 255
kernelSideLen = 10

printFileName = False
investigateCO = False
investigateMC = False
testClose = True
dispSeparateAcc = False
adjustBinThre = False

overallM = 0
overallC = 0

gazeDirection = -1

kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelSideLen, kernelSideLen))

def nothing(x):
    pass


def standardize(img, close, binThre):
    img = cv.resize(img,(256,256))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binImg = cv.threshold(gray, binThre, highAbsThre, cv.THRESH_BINARY)
    if close:
        binImg = cv.morphologyEx(binImg, cv.MORPH_CLOSE, kernel)
    else:
        binImg = cv.morphologyEx(binImg, cv.MORPH_OPEN, kernel)
    return cv.Canny(binImg, 30, 40)


def computePercent(suffix, binThre):
    global investigateMC, investigateCO, overallM, overallC
    # print(suffix)
    if(suffix == 'L'):
        gazeDirection = 0
    elif(suffix == 'M'):
        gazeDirection = 1
    elif(suffix == 'R'):
        gazeDirection = 2

    inputFolder = sessionName + suffix
    path = './imgs/' + inputFolder

    fileList = os.listdir(path)

    totalRecs = int(len(fileList) / 2)
    correctCircle = 0
    correctMass = 0

    # print(fileList)

    # print("starting with totalRecs = " + str(totalRecs))
    skipMC = not investigateMC

    for i in range(totalRecs):
        if printFileName:
            print(str(i+1) + ">", end='')
        if printFileName:
            print(fileList[2 * i] + " | " + fileList[2 *i + 1])

        samp0 = cv.imread(path + '/' + fileList[2 * i])
        samp1 = cv.imread(path + '/' + fileList[2 * i + 1])

        img0 = standardize(samp0, testClose, binThre)
        img1 = standardize(samp1, testClose, binThre)

        samp0 = cv.resize(samp0, (256, 256))
        samp1 = cv.resize(samp1, (256, 256))

        contours0, hierarchy0 = cv.findContours(img0, 1, 2)
        contours1, hierarchy1 = cv.findContours(img1, 1, 2)

        biggestContourIndex0, biggestContourIndex1 = 0, 0

        if len(contours0) == 0 or len(contours1) == 0:
            continue

        # for i in range(1, len(contours0)):
        #     if cv.contourArea(contours0[i]) > \
        #             cv.contourArea(contours0[biggestContourIndex0]):
        #         biggestContourIndex0 = i

        # for i in range(1, len(contours1)):
        #     if cv.contourArea(contours1[i]) > \
        #             cv.contourArea(contours1[biggestContourIndex1]):
        #         biggestContourIndex1 = i

        # M0 = cv.moments(contours0[biggestContourIndex0])
        # M1 = cv.moments(contours1[biggestContourIndex1])

        # try:
        #     cMx0 = int(M0["m10"] / M0["m00"])
        #     cMy0 = int(M0["m01"] / M0["m00"])
        #     cMx1 = int(M1["m10"] / M1["m00"])
        #     cMy1 = int(M1["m01"] / M1["m00"])
        # except ZeroDivisionError:
        #     continue

        # massIndex = cMx0 + cMx1

        (xCNT0, yCNT0), radius0 = cv.minEnclosingCircle(contours0[biggestContourIndex0])
        (xCNT1, yCNT1), radius1 = cv.minEnclosingCircle(contours1[biggestContourIndex1])

        xCNT0, yCNT0, xCNT1, yCNT1, radius0, radius1 = int(xCNT0), int(yCNT0), int(xCNT1), int(yCNT1), int(radius0), int(radius1)

        circleIndex = xCNT0 + xCNT1

        if not skipMC:
            cv.circle(img0, (xCNT0, yCNT0), radius0, (255, 255, 255), 1)
            cv.circle(img1, (xCNT1, yCNT1), radius1, (255, 255, 255), 1)

            cv.line(img0, (xCNT0, yCNT0 + radius0), (xCNT0, yCNT0 - radius0), (255, 255, 255), 1)
            cv.line(img0, (xCNT0 + radius0, yCNT0), (xCNT0 - radius0, yCNT0), (255, 255, 255), 1)
            cv.line(img1, (xCNT1, yCNT1 + radius1), (xCNT1, yCNT1 - radius1), (255, 255, 255), 1)
            cv.line(img1, (xCNT1 + radius1, yCNT1), (xCNT1 - radius1, yCNT1), (255, 255, 255), 1)

            cv.line(samp0, (xCNT0, yCNT0 + radius0), (xCNT0, yCNT0 - radius0), (255, 255, 0), 1)
            cv.line(samp1, (xCNT1, yCNT1 + radius1), (xCNT1, yCNT1 - radius1), (255, 255, 0), 1)

            # cv.line(img0, (cMx0, cMy0 + 8), (cMx0, cMy0 - 8), (255, 255, 255), 2)
            # cv.line(img0, (cMx0 + 8, cMy0), (cMx0 - 8, cMy0), (255, 255, 255), 2)
            # cv.line(img1, (cMx1, cMy1 + 8), (cMx1, cMy1 - 8), (255, 255, 255), 2)
            # cv.line(img1, (cMx1 + 8, cMy1), (cMx1 - 8, cMy1), (255, 255, 255), 2)

            # cv.line(samp0, (cMx0, cMy0 + 8), (cMx0, cMy0 - 8), (0, 255, 255), 2)
            # cv.line(samp1, (cMx1, cMy1 + 8), (cMx1, cMy1 - 8), (0, 255, 255), 2)

            while True:
                cv.imshow("samp0 | samp1", np.hstack((samp0, samp1)))
                cv.imshow("img0 | img1", np.hstack((img0, img1)))
                keyValue = cv.waitKey(20)
                if keyValue & 0xff == ord('n'):
                    break
                if keyValue & 0xff == ord('x'):
                    skipMC = True
                    break

        if investigateCO:
            while True:
                cv.imshow("CLOSE | OPEN", np.vstack((np.hstack((standardize(samp0, True), standardize(samp0, False))),np.hstack((standardize(samp1, True), standardize(samp1, False))))))
                keyValue = cv.waitKey(20)
                if keyValue & 0xff == ord('n'):
                    break
                if keyValue & 0xff == ord('x'):
                    investigateCO = False
                    break

        if gazeDirection == 0:
            # if massIndex > lookLeftThreshold:
            #     correctMass += 1
            if circleIndex > lookLeftThreshold:
                correctCircle += 1
        if gazeDirection == 1:
            # if massIndex < lookLeftThreshold and massIndex > lookRightThreshold:
            #     correctMass += 1
            if circleIndex < lookLeftThreshold and circleIndex > lookRightThreshold:
                correctCircle += 1
        if gazeDirection == 2:
            # if massIndex < lookRightThreshold:
            #     correctMass += 1
            if circleIndex < lookRightThreshold:
                correctCircle += 1
    # print("done")

    if dispSeparateAcc:
        print("Test complete in folder " + str(inputFolder) + " with " + str(totalRecs) +
              " instances | lowBinThresh = " + str(lowAbsThre) + " | testClose = " + str(testClose))
        # print("Moment Calculation Accuracy = \t\t" + str((correctMass/totalRecs*100)) + "%")
        print("Circle Fit Calculation Accuracy = \t" + str((correctCircle / totalRecs * 100)) + "%\n")
    #overallM += (correctMass/totalRecs*100)
    overallC += (correctCircle/totalRecs*100)

    cv.destroyAllWindows()


def computeAll(threshold):
    suffixes = ['L', 'M', 'R']
    for suf in suffixes:
        computePercent(suf, threshold)


sessionName = input("enter input session name > ")

if adjustBinThre:
    print("Adjust binary thresholds...")

    cv.namedWindow('thresh')
    cv.createTrackbar('binLo', 'thresh', lowAbsThre, 255, nothing)
    cv.createTrackbar('binHi', 'thresh', highAbsThre, 255, nothing)

    path = './imgs/' + sessionName + 'M'
    sampleName = os.listdir(path)[0]

    while True:
        cv.imshow('thresh', np.hstack((standardize(cv.imread(path + '/' + sampleName), True),standardize(cv.imread(path + '/' + sampleName), False))))
        lowAbsThre = cv.getTrackbarPos('binLo', 'thresh')
        highAbsThre = cv.getTrackbarPos('binHi', 'thresh')
        keyValue = cv.waitKey(20)
        if keyValue & 0xff == ord('c'):
            break

    cv.destroyAllWindows()

points = []

for thre in range(30, 120):
    computeAll(thre)
    # print("Overall Moment Accuracy = \t\t\t" + str((overallM/3)) + "%")
    print("Computation Complete with Thre @ " + str(thre) + " | ", end="")
    print("Overall Accuracy = \t\t" + str((overallC / 3)) + "%")
    points.append([thre, (overallC / 3)])
    overallC = 0

x, y = np.array(points).T
plt.scatter(x, y)
plt.xlabel("Binary Min Threshold")
plt.ylabel("% Accuracy of Set = " + sessionName)
plt.show()




