#import libraries
import cv2
import numpy as np


# Create a VideoCapture object to open webcam
cap = cv2.VideoCapture(0)
# See hand detection in the video
hand_cascade = cv2.CascadeClassifier('hand03.xml')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.video.isOpened():
            rval, frame = self.video.read()
            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = facec.detectMultiScale(gray_fr, 1.3, 5)
            hands = hand_cascade.detectMultiScale(gray_fr, 1.3, 5)    
        else:
            rval = False

        while rval:
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]
#                 roi = cv2.resize(fc, (48, 48))
                roi = cv2.resize(fc, (224, 224))
                # pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
#                 pred = model.predict_number(roi[np.newaxis, :, :, 3])                
                pred = "unknown"
                cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)    
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()