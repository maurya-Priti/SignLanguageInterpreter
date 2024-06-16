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

# def gen_frames():

#   cap = cv2.VideoCapture(0)

#   hands = mp_hands.Hands(
#       min_detection_confidence=0.7, min_tracking_confidence=0.5)

#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = hands.process(image)
    
#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     if results.multi_hand_landmarks:
#       points=[]
#       for hand_landmarks in results.multi_hand_landmarks:
#         for i in range(21):
#             x=hand_landmarks.landmark[i].x
#             y=hand_landmarks.landmark[i].y
#             points.append(x)
#             points.append(y)  
#       # print(points)
#       if len(points)==42:
#               predict=forrest_model.predict([points])
#       #print(predict)    
              
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     if results.multi_hand_landmarks:
          
#         cv2.putText(image, '%s' % (predict), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
      
#     # cv2.imshow('MediaPipe Hands', image)
#     # if cv2.waitKey(5) & 0xFF == 27:
#     #   break

#     ret, buffer = cv2.imencode('.jpg', image)
#     image = buffer.tobytes()
#     yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')







if __name__ == '__main__':
    app.run(debug=True)
