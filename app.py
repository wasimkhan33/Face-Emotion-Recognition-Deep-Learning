# Importing required libraries, obviously
import logging
import logging.handlers
import threading
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from typing import Union



try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
    


# Loading pre-trained parameters for the cascade classifier
try:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Face Detection
    classifier =load_model('model.h5')  #Load model
    emotion_labels = ['Angry','Disgust','Fear','Happy','Sad', 'Surprise', 'Neautral']  # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")
    
    
class VideoTransformer(VideoTransformerBase):
    
    

    def transform(self, frame):
        label=[]
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        
        
        


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3,1)
        

        for (x,y,w,h) in faces:
            a=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0) ## reshaping the cropped face image for prediction
            prediction = classifier.predict(roi)[0]   #Prediction
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
               
        return b

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)





WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
    
    


def about():
	st.write(
		'''

		Face Emotion Detection App. It can be used to detect objects in images or videos.
        
        Facial recognition systems can be used to identify people in photos, videos, 
        or in real-time. Facial recognition is a category of biometric security. Other
        forms of biometric software include voice recognition, fingerprint recognition,
        and eye retina or iris recognition.

Read more :
    For Cascade Classification: 
        https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
		''')

def main():
    
    activities = ["Home","Real-Time Face Detection","About","Contact Us"]
    choice = st.sidebar.selectbox("Select from drop down", activities)
    
        
    if choice =="Real-Time Face Detection":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Detection WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write("Go to the About section from the sidebar to learn more about it.")
        st.write("**Instructions while using the APP**")
        st.write('''
                  
                  1. Select WebCam if your system haing multiple webcam/software

                  2. Click on the Start button to start.

                  3. Webcam will open automatically and predict at that instant.
                                    
                  4. Click on  Stop  to end.''')
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        
      
        
    elif choice == "About":
        
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Haar Cascade Object Detection</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        about()
    elif choice=="Contact Us":
        with st.form(key='my_form'):
            text_input = st.text_input(label='Enter Query')
            submit_button = st.form_submit_button(label='Submit')
        st.write('''
                  Email:- waseem3378@gmail.com                 
                  ''')
        
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:0.25px">
    <h2 style="color:white;text-align:center;">Thanks you</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
    elif choice=="Home":
         html_temp = """
    <body style="background-color:red;">
    <div style="background-image: url('https://images.unsplash.com/photo-1460602594182-8568137446ce?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxhbGx8fHx8fHx8fHwxNjIwMzk3ODcy&ixlib=rb-1.2.1&q=80&w=1080&utm_source=unsplash_source&utm_medium=referral&utm_campaign=api-credit');padding:150px">
    <h2 style="color:MintCream;text-align:center;"><b>HI EVERYONE.</b></h2>
    <h2 style="color:FloralWhite;text-align:center;">SIT STRAIGHT YOU ARE UNDER FACE DETECTION WEBAPP.</h2>
    </div>
    </body>
        """
         st.markdown(html_temp, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
