import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import mediapipe as mp

from sklearn.metrics import accuracy_score


from flask import Flask, request, jsonify,Response,render_template
import base64
from PIL import Image
import io

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')




x_cor=pd.read_csv("ASLcoordinates.csv")
y=x_cor.Letter
x = np.array(x_cor.drop(["Letter"],axis=1))





forrest_model = RandomForestClassifier(random_state=1)







forrest_model.fit(x,y)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



@app.route('/process_frame', methods=['POST'])
def process_frame():
    hands = mp_hands.Hands(
      min_detection_confidence=0.7, min_tracking_confidence=0.5)
    try:
        frame_data = request.form['frame_data']
        image_data = base64.b64decode(frame_data.split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Process the frame

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
  
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
        if results.multi_hand_landmarks:
          points=[]
          for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                points.append(x)
                points.append(y)  
          # print(points)
          if len(points)==42:
                  predict=forrest_model.predict([points])
          print(predict)    
            
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if results.multi_hand_landmarks:
              
            cv2.putText(image, '%s' % (predict), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
       


        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image_data': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500








if __name__ == '__main__':
    app.run()
