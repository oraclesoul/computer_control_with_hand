import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyautogui

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300


labels = ["A","B","C"]

# pyautogui.click(300,300)
# pyautogui.press('capslock')

while 1:
    
    success, img = cap.read()
    imgOutput = img.copy()
    imgOutputHeight,imgOutputWidth,_ = imgOutput.shape
    # print("Image shize",imgOutputHeight," ",imgOutputWidth)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # when some portion of hand goes out of camera
        # if x<100 | y<100 | x+w>540 | y+h > 380:
        #     print("hand position is out of box: ",x," ",y)
        #     continue

        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
        imgCropH,imgCropW,_ = imgCrop.shape
        print("share of imagecropeshare",imgCropH,imgCropW)

        if imgCropH==0 or imgCropW ==0 : continue

        aspectRatio = imgCropH / imgCropW

        if aspectRatio > 1:
            k = imgSize / imgCropH
            wCal = math.ceil(k * imgCropW)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)

        else:
            k = imgSize / imgCropW
            hCal = math.ceil(k * imgCropH)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                    (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                    (x + w+offset, y + h+offset), (255, 0, 255), 4)


        if index==2:
            # pyautogui.press('win',1)
            pyautogui.keyDown('winleft')
            pyautogui.press('d')
            pyautogui.keyUp('winleft')
            print("index 2")
        elif index==0:
            print("index 0")
        else:
            pyautogui.press('q')

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    q = cv2.waitKey(1)
    if(q == ord('q')):
        break
