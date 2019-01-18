#importing all files
from imutils import face_utils
import numpy as np
import imutils
import dlib
import math
import cv2

#facial landmarks key points 
faciallandmarks =[
	("jaw_keypoints", (0, 17)),
	("righteyebrow_keypoints", (17, 22)),
	("lefteyebrow_keypoints", (22, 27)),
	("nose_keypoints", (27, 36)),
	("righteye_keypoints", (36, 42)),
	("lefteye_keypoints", (42, 48)),
	("mouth_keypoints", (48, 68))
	
]

#rectangle points
def rectbb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

#numpy array conversion
def shapenp(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

#front face dectetor
detect = dlib.get_frontal_face_detector()
#key points predictor
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#video capturing
cam=cv2.VideoCapture(0)

#yawn detection
def yawn_detection(shape,dtype="int"):
    top_lips=[]
    bottom_lips=[]
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if 50<=i<=53 or 61<=i<=64:
            top_lips.append(coords[i])
        
        elif 65<=i<=68 or 56<=i<=59:
            bottom_lips.append(coords[i])
        toplipsall=np.squeeze(np.asarray(top_lips))
        bottomlipsall=np.squeeze(np.asarray(bottom_lips))
        top_lips_mean=np.array(np.mean(toplipsall,axis=0),dtype=dtype)
        bottom_lips_mean=np.array(np.mean(bottomlipsall,axis=0),dtype=dtype)
        top_lips_mean = top_lips_mean.reshape(-1) 
        bottom_lips_mean=bottom_lips_mean.reshape(-1) 
        
        #distance=math.sqrt((bottom_lips_mean[0] - top_lips_mean[0])**2 + (bottom_lips_mean[-1] - top_lips_mean[-1])**2)
        distance=bottom_lips_mean[-1] - top_lips_mean[-1]
        
        yawn=False
        
        if distance>20:
            yawn=True
            print(yawn)
    
    return yawn


#pout and smile detection
def pout_detection(shape,dtype="int"):
    left_corner=[]
    right_corner=[]
    coords = np.zeros((68, 2), dtype=dtype)
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
        if i==65 or i==55:
            right_corner.append(coords[i])
        
        elif i==60 or i==48:
            left_corner.append(coords[i])
        leftcornerall=np.squeeze(np.asarray(left_corner))
        rightcornerall=np.squeeze(np.asarray(right_corner))
        leftmean=np.array(np.mean(leftcornerall,axis=0),dtype=dtype)
        rightmean=np.array(np.mean(rightcornerall,axis=0),dtype=dtype)
        leftmean_flat = leftmean.reshape(-1) 
        rightmean_flat=rightmean.reshape(-1)
        distance=abs(leftmean_flat[0]-rightmean_flat[0])  
        pout=False
        smile=False
        
        if distance<35:
            pout=True
        
        elif distance>41:
            smile=True
    
    return (pout,smile)
        
#frame capturing
while True:
    a,image = cam.read()
    image = cv2.resize(image,(600,500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 1)
    
    for (index, rect) in enumerate(rects):
        shape = predict(gray, rect)
        shape_new = shapenp(shape)
        (x, y, w, h) =rectbb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face No.{}".format(index + 1), (x - 8, y - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        count=0
        
        for (x, y) in shape_new:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(image,"{}".format(count + 1), (x - 1, y - 1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),1)
            count+=1
    
    value=yawn_detection(shape)
    if value==True:
        cv2.putText(image,"Yawning",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),2)
    
    pout,smile=pout_detection(shape)
    
    if pout==True:
        cv2.putText(image,"Pout",(50,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),2)
    
    if smile==True:
        cv2.putText(image,"Smile",(50,70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),2)
    
    cv2.putText(image,"Press q to exit",(480,440),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (244, 0, 255),2)
    
    cv2.imshow("result", image)
    
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        break

#webcam release
cam.release()
#all windows destroying
cv2.destroyAllWindows()

