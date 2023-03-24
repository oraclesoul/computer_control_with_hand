# RUN VSCODE AS ADMIN
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyautogui

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 10
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

wScr, hScr = pyautogui.size()
print(wScr, hScr)

offset = 20
imgSize = 300

labels = ["A","B","C"]

while 1:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
            hand = hands[0]
            if hand['type'] == "Left":
                x, y, w, h = hand['bbox']
                imgOutput = img.copy()
                imgOutputHeight,imgOutputWidth,_ = imgOutput.shape
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape
                imgCropH,imgCropW,_ = imgCrop.shape

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
                    pyautogui.keyDown('winleft')
                    pyautogui.press('d')
                    pyautogui.keyUp('winleft')
                    print("index 2")
                elif index==0:
                    pyautogui.leftClick()
                    print("index 0")
                else:
                    pyautogui.hotkey('ctrl','win','o')
                    print("index 1")
                cv2.imshow("Image", imgOutput)
            else:
                print("Right Hand")
                lmList = hand['lmList']
                print(lmList[8])
                if len(lmList) != 0:
                    x1, y1,_ = lmList[8][0:]
                    x2, y2,_ = lmList[12][0:]
                    print(x1, y1, x2, y2)
                
                # 3. Check which fingers are up
                fingers = detector.fingersUp(hand)
                # print(fingers)
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                (255, 0, 255), 2)
                # 4. Only Index Finger : Moving Mode
                if fingers[1] == 1 and fingers[2] == 0:
                    # 5. Convert Coordinates
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    # 6. Smoothen Values
                    print("x3,y3",x3,y3)
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    
                    # 7. Move Mouse
                    pyautogui.moveTo(wScr-clocX, clocY)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    plocX, plocY = clocX, clocY
                    
                # 8. Both Index and middle fingers are up : Clicking Mode
                if fingers[1] == 1 and fingers[2] == 1:
                        pyautogui.click()
                
                # 11. Frame Rate
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
                cv2.imshow("Mouse",img)    
    q = cv2.waitKey(1)
    if(q == ord('q')):
        break
# RUN VS CODE AS ADMIN
