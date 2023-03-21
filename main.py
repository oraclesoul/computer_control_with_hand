import cv2 

#Parameters
width, height = 360, 360

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    cv2.imshow("Camera",img)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break