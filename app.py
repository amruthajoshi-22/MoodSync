from flask import Flask,render_template,Response,request
import numpy as np
import cv2
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from time import sleep


app=Flask(__name__)

MODEL_PATH='models/model.h5'

''' import os
if os.environ.get('WERKZEUG_RUN_MAIN'):
    camera = cv2.VideoCapture(0)
    '''
camera = cv2.VideoCapture(0)
#label='Happy'
def detect_emotion(frame,emotion_labels,face_classifier,classifier,labels):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            global label 
            label= emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    return frame


def generate_frames():
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classifier = load_model(MODEL_PATH)
    emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    while True:
        success,frame=camera.read()
        labels=[]
        if not success:
            break
        else:
            frame=detect_emotion(frame,emotion_labels,face_classifier,classifier,labels)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
    return frame 

@app.route('/')
def index():
    return render_template('video.html') 

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def emotion():
    camera.release()
    return render_template('song.html',label=label)

    

if __name__=="__main__":
    app.run(debug=True)
