import cv2
import numpy as np 
from matplotlib import pyplot as plot
def onmouse(event, x, y, flags, param):   
  if event==cv2.EVENT_MOUSEMOVE:      
    print("mouse_position:%d,%d" % (x,y)) 

cap = cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('G:/anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
cv2.namedWindow("mouse_position")
cv2.setMouseCallback("mouse_position",onmouse)
while(1):
    ret,frame=cap.read()
    cv2.imshow("capture",frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,h,w) in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print("face position:x=%d,y=%d,h=%d,w=%d" %(x,y,h,w))
    cv2.imshow("faces",frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()