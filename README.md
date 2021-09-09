# FACE-EMOTION-RECOGNITION

## Author

- [@Wasim khan](https://github.com/wasimkhan33/)

  
## Problem Statement 

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (ex- Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to a lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

We will solve the above-mentioned challenge by applying deep learning algorithms to live video data.The solution to this problem is by recognizing facial emotions.
## Dataset Information

The data comes from the past Kaggle competition “Challenges in Representation Learning: Facial Expression Recognition Challenge”:
we have defined the image size to 48 so each image will be reduced to a size of 48x48.The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. Each image corresponds to a facial expression in one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The dataset contains approximately 36K images.

Dataset link - [Click here](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset)

## Emotion detection Recognition using deep learning

CNN Model

Classic NNs are usually composed of several fully connected layers. This means that every node of one layer is connected to all the nodes of the next layer.
Convolutional Neural Networks also have Convolutional layers that apply sliding functions to groups of pixels that are next to each other. 


![](https://github.com/wasimkhan33/Deep-Learning-Capstone-Face-Emotion-Recognition/blob/main/IMG/1-4.png?raw=true)

There are two main parts to a CNN architecture-
A convolution tool that separates and identifies the various features of the image for analysis in a process called as Feature Extraction
A fully connected layer that utilizes the output from the convolution process and predicts the class of the image based on the features extracted in previous stages.

This was the model structure. In the output layer there were 7 nodes. This model was used to predict emotion in following ways:

![](https://github.com/wasimkhan33/Deep-Learning-Capstone-Face-Emotion-Recognition/blob/main/IMG/emoji11'.png?raw=true)

First, the haar cascade method is used to detect faces in each frame of the webcam feed.
The region of image containing the face is resized to 48x48 and is passed as input to the CNN.
The network outputs a list of softmax scores for the seven classes of emotions.
The emotion with maximum score is displayed on the screen.
## Dependencies

- Python
- Tensorflow
- Keras
- Opencv
- Streamlit
- Streamlit-Webrtc

  
## Deployment

Deployment done for this project on Heroku and Streamlit share

- Deployment Link for Heroku - [Click here](https://faceemotiondetection-wasim.herokuapp.com/) 
- Deployment Link for Streamlit Share - [Click here](https://share.streamlit.io/wasimkhan33/face-emotion-detection-using-streamlit/main/app.py)
- Face emotion recognition using streamlit - [Click here](https://github.com/wasimkhan33/face-emotion-detection-using-streamlit.git)


  
## Installation

Run project with

```Software
  Anaconda
  Jupyter notebook
  Visual Studio Code
```
    
## Run Locally

Clone the project

```bash
  git clone https://github.com/wasimkhan33/Deep-Learning-Capstone-Face-Emotion-Recognition.git
```

Open Anaconda Prompt &
Go to the project directory
```bash
  cd Deep-Learning-Capstone-Face-Emotion-Recognition
```

Install dependencies

```bash
  pip install -r requirement.txt
```

Start local webcam

```bash
  python camera.py
```

  
## Demo

![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/55997315/132719259-80efe535-054c-4f16-8503-6af0488c45b2.gif)


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/waseem3378/)
[![twitter](https://img.shields.io/badge/github-211F1F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wasimkhan33)

  
