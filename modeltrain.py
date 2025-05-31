#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import socket
import pandas as pd
from joblib import load
#import pyttsx3
import time
from sklearn.preprocessing import MinMaxScaler
import struct 
from sklearn.metrics import accuracy_score
import logging
import subprocess


# Load the trained model and scaler
model = load('gesture_recognition_model.pkl')
scaler = load('scaler.pkl')
le = load('labelencoder.pkl')

#engine = pyttsx3.init()

# ESP32 IP address and port
#HOST = '10.206.1.250'
HOST = '0.0.0.0'
PORT = 4444

# Function to receive sensor data from ESP32
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            #tmp=0
            while True:
                data = conn.recv(44)
                print("Connection received\n")
               
                if not data:
                    break
                
                while len(data) < 44:
                        packet = conn.recv(44 - len(data))
                        if not packet:
                                break
                        data += packet
                #print(f"Raw data: {data}\n")
                if len(data) == 44:
                        angles = struct.unpack("5f", data[:20])# Assuming 3 floats for accelerometer
                        gyro = struct.unpack('3f', data[20:32]) 
                        accel = struct.unpack('3f', data[32:])
                        


                     
                      
                    
                        angles_dict = {'flex_{}'.format(i+1): angle for i, angle in enumerate(angles)}
                        gyro_dict = {'GYR{}'.format(i+1): gyro_value for i, gyro_value in enumerate(gyro)}
                        accel_dict = {'ACC{}'.format(i+1): accel_value for i, accel_value in enumerate(accel)}


                        df_angles = pd.DataFrame([angles_dict])
                        df_gyro = pd.DataFrame([gyro_dict])
                        df_accel = pd.DataFrame([accel_dict])


                        df = pd.concat([df_angles,  df_gyro ,  df_accel], axis=1)
                    
                        df.columns = ['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5', 'GYRx', 'GYRy', 'GYRz', 'ACCx', 'ACCy', 'ACCz' ]




                   



                      
                        scaled_data = scaler.transform(df)
                        predicted_label = le.inverse_transform(model.predict(scaled_data))
                        predicted_alphabet = predicted_label[0]
                        print(f"Predicted alphabet: {predicted_alphabet}")
                        subprocess.run(['espeak', predicted_alphabet])
                       

                        




                        time.sleep(1)  # Adjust the delay as needed
                        
                        #tmp=tmp+1

