import math

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

#video intializer
cap = cv2.VideoCapture('Videos/vid (7).mp4')

#color intializer
colorfinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 120, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

#centre points
positionX, positionY = [], []
xlist = ([item for item in range(0, 1300)])
start = True
prediction = False

while True:
    if start:
        if len(positionX) == 10:start=False
        #video input
        suc, vid = cap.read()
        vid = vid[0:900, :]

        # image imput
        # img = cv2.imread('Ball.png')
        # img = img[0:900, :]

        #To find color of ball
        ballcolor, mask = colorfinder.update(vid, hsvVals)

        #To get ball location
        imgContours, contours = cvzone.findContours(vid, mask, minArea=500)
        if contours:
            positionX.append(contours[0]['center'][0])
            positionY.append(contours[0]['center'][1])
            # print(position)

        if positionX:

            ### Predicting The Path Using Polynomial Regression y = Ax^2 + Bx + C
            #finding the coefficient
            A, B, C = np.polyfit(positionX, positionY, 2)

            for i, (pointX, pointY) in enumerate(zip(positionX, positionY)):
                point = (pointX, pointY)
                cv2.circle(imgContours, point, 5, (0, 255, 0), cv2.FILLED)
                ##to draw line
                if i == 0:
                    cv2.line(imgContours, point, point, (0, 255, 0), 3)
                else:
                    cv2.line(imgContours, point, (positionX[i - 1], positionY[i - 1]), (0, 255, 0), 3)

            for x in xlist:
                y = int(A*x**2 + B*x + C)
                cv2.circle(imgContours, (x, y), 1, (255, 0, 255), cv2.FILLED)


            if len(positionX) < 10:
                ###PREDICTION
                #x = 320 to 435, y = 590
                a = A
                b = B
                c = C - 590

                x = int(((-b - math.sqrt(b**2 - (4*a*c))))/(2*a))
                prediction = 320 < x < 435

            if prediction:
                cv2.rectangle(imgContours, (60, 35), (300, 125), (0, 200, 0), thickness=-1)
                cv2.putText(imgContours, 'Basket', (75, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)
                print('Basket')
            else:
                cv2.rectangle(imgContours, (50, 35), (400, 125), (0, 0, 255), thickness=-1)
                cv2.putText(imgContours, 'No Basket', (65, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)
                print('No Basket')


        #display section
        ## for video
        video = cv2.resize(imgContours, (800, 800))
        cv2.imshow('Video', video)

    key = cv2.waitKey(100)
    if key == ord('s'):
        start = True
    if key == ord('q'):
        break
