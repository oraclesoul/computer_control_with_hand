# RUN VSCODE AS ADMIN
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyautogui

video = cv2.VideoCapture(0)
handDetector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

##########################
cam_width, cam_height = 640, 480
frame_rate = 100 # Frame Reduction
smoothening_value = 10
screen_width, screen_height = pyautogui.size()
# print(screen_width, screen_height)
margin = 20
desired_imageSize = 300
#########################

prev_locX, prev_locY = 0, 0
curr_locX, curr_locY = 0, 0

labels = ["backspace","Show_desktop","Show_On_screen_keyboard","Switch_App","StartSlideShow","Next","EndShow","Previous"]

gestureTime = time.time()
leftClickTime = time.time()

while 1:
    _, inputImage = video.read()
    hands, inputImage = handDetector.findHands(inputImage)
    if hands:
            hand = hands[0]
            if hand['type'] == "Left":
                x, y, w, h = hand['bbox']
                output_image = inputImage.copy()
                imgOutputHeight,imgOutputWidth,_ = output_image.shape
                finalDataSetImage = np.ones((desired_imageSize, desired_imageSize, 3), np.uint8) * 255
                croppedInputImage = inputImage[y - margin:y + h + margin, x - margin:x + w + margin]

                
                croppedInputImageH,croppedInputImageW,_ = croppedInputImage.shape

                if croppedInputImageH==0 or croppedInputImageW ==0 : continue

                H_W_Ratio = croppedInputImageH / croppedInputImageW

                if H_W_Ratio > 1:
                    k = desired_imageSize / croppedInputImageH
                    calculated_width = math.ceil(k * croppedInputImageW)
                    resizedImage = cv2.resize(croppedInputImage, (calculated_width, desired_imageSize))
                    
                    wGap = math.ceil((desired_imageSize - calculated_width) / 2)
                    finalDataSetImage[:, wGap:calculated_width + wGap] = resizedImage
                    prediction, index = classifier.getPrediction(finalDataSetImage, draw=False)

                else:
                    k = desired_imageSize / croppedInputImageW
                    calculated_height = math.ceil(k * croppedInputImageH)
                    resizedImage = cv2.resize(croppedInputImage, (desired_imageSize, calculated_height))
                    
                    hGap = math.ceil((desired_imageSize - calculated_height) / 2)
                    finalDataSetImage[hGap:calculated_height + hGap, :] = resizedImage
                    prediction, index = classifier.getPrediction(finalDataSetImage, draw=False)

                cv2.rectangle(output_image, (x - margin, y - margin-50),
                            (x - margin+90, y - margin-50+50), (255, 0, 255), cv2.FILLED)
                cv2.putText(output_image, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(output_image, (x-margin, y-margin),
                            (x + w+margin, y + h+margin), (255, 0, 255), 4)
                if time.time()-gestureTime>1.5:
                    match index:
                         case 1:
                              pyautogui.keyDown('winleft')
                              pyautogui.press('d')
                              pyautogui.keyUp('winleft')
                         case 2:
                              pyautogui.hotkey('ctrl','win','o')                              
                         case 3:
                              pyautogui.hotkey('alt','tab')
                         case 4:
                              pyautogui.hotkey('f5')
                         case 5:
                              pyautogui.hotkey('right')
                         case 6:
                              pyautogui.hotkey('esc')
                         case 7:
                              pyautogui.hotkey('left')
                         case _:
                              ()
                    gestureTime = time.time()

                cv2.imshow("Image", output_image)
            else:
                print("Right Hand")
                lmList = hand['lmList']
                # print(lmList[8])
                if len(lmList) != 0:
                    x1, y1,_ = lmList[8][0:]
                    x2, y2,_ = lmList[12][0:]
                    
                
                # 3. Check which rightHandFingers are up
                rightHandFingers = handDetector.fingersUp(hand)
                # print(rightHandFingers)
                cv2.rectangle(inputImage, (frame_rate, frame_rate), (cam_width - frame_rate, cam_height - frame_rate),
                (255, 0, 255), 2)
                # 4. Only Index Finger : Moving Mode
                if rightHandFingers[1] == 1 and rightHandFingers[2] == 0:
                    # 5. Convert Coordinates
                    x3 = np.interp(x1, (frame_rate, cam_width - frame_rate), (0, screen_width))
                    y3 = np.interp(y1, (frame_rate, cam_height - frame_rate), (0, screen_height))
                    # 6. Smoothen Values
                    
                    curr_locX = prev_locX + (x3 - prev_locX) / smoothening_value
                    curr_locY = prev_locY + (y3 - prev_locY) / smoothening_value
                    
                    # 7. Move Mouse
                    pyautogui.moveTo(screen_width-curr_locX, curr_locY)
                    cv2.circle(inputImage, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    prev_locX, prev_locY = curr_locX, curr_locY
                    
                # 8. Both Index and middle rightHandFingers are up : Clicking Mode
                if rightHandFingers[1] == 1 and rightHandFingers[2] == 1:
                        if time.time() - leftClickTime > 0.2:
                            pyautogui.click()
                            leftClickTime = time.time()
                
                cv2.imshow("Mouse",inputImage)
 
    q = cv2.waitKey(1)
    if(q == ord('q')):
        break
# RUN VS CODE AS ADMIN
