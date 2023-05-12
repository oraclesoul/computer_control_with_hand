import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

video = cv2.VideoCapture(0)
handDetector = HandDetector(maxHands=1)

margin = 20
desired_imageSize = 300

dataSet_location = "Images/G3"
dataSet_count = 0

while True:
    _, inputImage = video.read()
    hands, inputImage = handDetector.findHands(inputImage)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

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

        else:
            k = desired_imageSize / croppedInputImageW
            calculated_height = math.ceil(k * croppedInputImageH)
            resizedImage = cv2.resize(croppedInputImage, (desired_imageSize, calculated_height))
            hGap = math.ceil((desired_imageSize - calculated_height) / 2)
            finalDataSetImage[hGap:calculated_height + hGap, :] = resizedImage

        cv2.imshow("ImageCrop", croppedInputImage)
        cv2.imshow("ImageWhite", finalDataSetImage)

    cv2.imshow("Image", inputImage)
    key_pressed = cv2.waitKey(1)
    if ord("s") == key_pressed:
        dataSet_count += 1
        cv2.imwrite(f'{dataSet_location}/Image_{time.time()}.jpg',finalDataSetImage)
        print(dataSet_count)