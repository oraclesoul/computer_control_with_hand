import cv2 
from cvzone.HandTrackingModule import HandDetector

#Parameters
width, height = 360, 360


detector = HandDetector(detectionCon=0.8, maxHands=2)

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img,flipType = False)  # with draw
    cv2.imshow("Camera",img)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break