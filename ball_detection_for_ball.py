#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from cvzone.SerialModule import SerialObject
import time

real_object_size = 19

known_distance = 24
focal_length = 1.50

arduino = SerialObject("COM3")


balltracker = load_model('D:\\balltracker.h5')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (800, 800))
    frame = cv2.flip(frame, 1)
    frame = frame[50:800, 50:800, :]
    height, width = frame.shape[:2]
    
    frame = cv2.circle(frame,(400,400),5,(0,255,0),3)
    cv2.line(frame,(0,400),(800,400),(255,255,255),3)
    cv2.line(frame,(400,0),(400,800),(255,255,255),3)
    cv2.line(frame,(350,0),(350,800),(30, 255, 255),3)
    cv2.line(frame,(450,0),(450,800),(30, 255, 255),3)
    cv2.line(frame,(0,350),(800,350),(30, 255, 255),3)
    cv2.line(frame,(0,450),(800,450),(30, 255, 255),3)
    
    frame_center = 400,400
   

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = balltracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        
        # Calculate bounding box coordinates
        start_point = tuple(np.multiply(sample_coords[:2], [750, 750]).astype(int))
        end_point = tuple(np.multiply(sample_coords[2:], [750, 750]).astype(int))
        
       

        # Draw bounding box
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)
        
        center_bbox_x = (start_point[0] + end_point[0]) // 2
        center_bbox_y = (start_point[1] + end_point[1]) // 2
        bbox_center = (center_bbox_x, center_bbox_y)  # Calculate the center of the bounding box
        
        
        vector_x = bbox_center[0] - frame_center[0]
        vector_y = bbox_center[1] - frame_center[1]

        # Calculate the angle using arctan2
        angle_rad = np.arctan2(vector_y, vector_x)  
        angle_deg = np.degrees(angle_rad)
        
        
        cv2.circle(frame, (center_bbox_x , center_bbox_y), 5, (0, 255, 0), -1)
        
        cv2.line(frame,(400,400),(center_bbox_x , center_bbox_y),(154,94,424),3)
        
        
        # Calculate perceived size of the object
        perceived_size = np.linalg.norm(sample_coords[2:] - sample_coords[:2])
        
        
        # Calculate distance using simple proportional relation (assuming known object size)
        distance = (real_object_size *focal_length) / perceived_size  

        
        if yhat[0] > 0.5:
            if -70 < angle_deg < 70:  
                arduino.sendData([2])
                print("arduino.sendData([2])")
#                 time.sleep(1)
                
                if distance > 70:
                    arduino.sendData([3])
                    print("arduino.sendData([3])")
                elif distance < 60:
                    arduino.sendData([4])
                    print("arduino.sendData([4])")
#                     time.sleep()
                
                else:
                    arduino.sendData([0])
                    print("arduino.sendData([0])")
#                     time.sleep(1)
                    
            elif angle_deg < -110 or angle_deg > 110: 
                arduino.sendData([1])
                print("arduino.sendData([1])")
#                 time.sleep(1)
                
                if distance > 70:
                    arduino.sendData([3])
                    print("arduino.sendData([3])")
                elif distance > 60:
                    arduino.sendData([4])
                    print("arduino.sendData([4])")
#                     time.sleep(1)

                
                else:
                    arduino.sendData([0])
                    print("arduino.sendData([0])")
#                     time.sleep(1)
                    
            else:
                if distance > 70:
                    arduino.sendData([3])
                    print("arduino.sendData([3])")
                    
                elif distance < 60:
                    arduino.sendData([4])
                    print("arduino.sendData([4])")
#                     time.sleep(1)
        
                else:
                    arduino.sendData([0])
                    print("arduino.sendData([0])")
#                     time.sleep(1)
                    
        else:
            arduino.sendData([0])
            print("arduino.sendData([0])")
             



        # Display the distance estimation on the frame
        cv2.putText(frame, f'Distance: {distance:.2f} cm', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'angle_deg: {angle_deg:.0f} degree', (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 , 255), 1, cv2.LINE_AA)
        

    cv2.imshow('ballTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


        

