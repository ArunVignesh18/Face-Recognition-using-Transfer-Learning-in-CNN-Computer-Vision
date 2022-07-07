from keras_facenet import FaceNet
embedder = FaceNet()

import matplotlib.image as mpimg 
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

'''

'''

def load_train_data(n):
  lst = []
  for i in range(1,n+1):
    filename = str(i)+'.jpg'
    x = mpimg.imread('Data/face reco/train/'+filename)
    x = cv2.resize(x,(160,160))
    lst.append(x)
  df = np.array(lst)
  y_train = pd.read_csv('Data/face reco/train/y_train.csv',header=None)
  y_train = y_train.values
  return df,y_train

def load_test_data(n):
  lst = []
  for i in range(1,n+1):
    filename = str(i)+'.jpg'
    x = mpimg.imread(filename)
    x = cv2.resize(x,(160,160))
    lst.append(x)
  df = np.array(lst)
  y_test = pd.read_csv('Data/face reco/Test/y_test.csv',header=None)
  y_test = y_test.values
  return df,y_test

  '''
Loading the data set 
'''
X_train_orig,y_train = load_train_data(44)

X_train = embedder.embeddings(X_train_orig)

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

# y_test

# model.score(X_test,y_test)

'''
To recognise face of a given input file with path and filename.
Returns one of the six classes if face is from the training data faces,
else 'Face not found !' message is returned .
'''

def predict(filename):
  x = mpimg.imread(filename)
  x = cv2.resize(x,(160,160))
  x = x.reshape(1,160,160,3)
  x = embedder.embeddings(x)
  if(max(model.predict_proba(x)[0])<0.25):         #0.25 is the threshold
    print("Face not found !")
    return
  else:
    return model.predict(x)

cam = cv2.VideoCapture(0)
counted = True
start = True

while start:
  ret, frame = cam.read()

  font = cv2.FONT_HERSHEY_SIMPLEX

  currentTime = datetime.now()
  currentSecs = currentTime.strftime("%S")

  if counted:
    a = int(currentSecs) + 10
    c = int(currentSecs)
    if a >= 60:
      a -= 60
    counted = False

  b = int(currentSecs) - c
  displaySecs = 10 - b
  if displaySecs >=60:
    displaySecs = displaySecs - 60
  if displaySecs != 0:

    cv2.putText(frame, 
                'Capturing Image In ' + str(displaySecs), 
                (100, 50), 
                font, 1, 
                (0,0,0), 
                2, 
                cv2.LINE_4)
  
  if displaySecs == 0:
    cv2.putText(frame, 
                'Image Captured', 
                (100, 50), 
                font, 1, 
                (0,0,0), 
                2, 
                cv2.LINE_4)
  
  if not ret:
    print("failed to grab frame")
    break

  cv2.imshow("test", frame)

  currentTime = datetime.now()
  secondsNow = currentTime.strftime("%S")

  ms = currentTime.strftime("%f")

  k = cv2.waitKey(1)

  if k%256 == 27:
    print("Escape hit, closing...")
    start = False

  if a == int(currentSecs):
    img_name = "1.jpg"
    cv2.imwrite(img_name, frame)
    X_test_orig,y_test = load_test_data(1)
    X_test = embedder.embeddings(X_test_orig)
    y_test
    model.score(X_test,y_test)
    groot = model.predict(X_test)
    print("Groot = ", groot)
    counted = True
    
cam.release()
