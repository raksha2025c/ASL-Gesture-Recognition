#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.tree import plot_tree
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report



#alphabets = [ 'a', 'b', 'c', 'd','e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']



#alphabets = [ 'f',  'h', 'i', 'j',  'l', 'n',  'p', 'w', 'x', 'y', 'z', 'hungry', 'hello', 'thankyou', 'goodbye', 'sorry']
#alphabets = ['bad','deaf','fine','good','goodbye','hello','hungry','me','no','please','sorry','thankyou','yes','you' ]
alphabets = [ 'a', 'b', 'c', 'd','e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'bad','deaf','fine','good','goodbye','hello','hungry','me','no','please' ]


merged_dfs = []

for alphabet in alphabets:
    file_path = f'/home/raksha/Downloads/modifieddataset/alphabet/{alphabet}_merged.csv_exported.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            merged_dfs.append(df)

combined_df = pd.concat(merged_dfs, ignore_index=True)
columns_to_filter = ['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5']
combined_df[columns_to_filter] = combined_df[columns_to_filter].where(combined_df[columns_to_filter] >= 0, 0)
combined_df.dropna(subset=columns_to_filter, inplace=True)


#combined_df.reset_index(drop=True, inplace=True)
X = combined_df.drop(['Alphabet', 'timestamp', 'user_id',
                      'Qw', 'Qx', 'Qy', 'Qz', 'ACCx_body',
                      'ACCy_body',
                      'ACCz_body',
                      'ACCx_world', 'ACCy_world', 'ACCz_world'], axis=1)
y = combined_df['Alphabet']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=10)


#model = RandomForestClassifier(n_estimators=600, random_state=15 , oob_score = True )
model = RandomForestClassifier(n_estimators= 2000, random_state=15 , oob_score = True , min_samples_leaf=5 ,   max_samples = 50 , max_depth = 5 , min_samples_split=4 , max_leaf_nodes = 150 )

#model = RandomForestClassifier(n_estimators=600, random_state=15 , oob_score = True ,  min_samples_leaf=5 ,   max_samples = 400 , max_depth = 6 , min_samples_split=6 , max_leaf_nodes = 150)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


oob_score = 1 -  model.oob_score_
print(f"Out-of-Bag Score: {oob_score}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



y_true = y_test
#y_pred = model.predict(X_test)



confusion_matrix = confusion_matrix(y_true, y_pred)

class_labels = alphabets
confusion_matrix_norm = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(20, 20))

cmap = sns.color_palette("Blues", as_cmap=True)


sns.heatmap(confusion_matrix_norm, annot=True, cmap=cmap, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)

dump(model, 'gesture_recognition_model.pkl')
dump(scaler, 'scaler.pkl')
dump(le, 'labelencoder.pkl')

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.show()


# In[3]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and scaler from .pkl files
#with open('gesture_recognition_model.pkl', 'rb') as model_file:
model = joblib.load('gesture_recognition_model.pkl')

#with open('scaler.pkl', 'rb') as scaler_file:
scaler = joblib.load('scaler.pkl')
labelencoder = joblib.load('labelencoder.pkl')

joblib.dump(model, 'gesture_recognition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(labelencoder, 'labelencoder.pkl')

# Hardcoded input values for a gesture (replace with real data)
input_data = {
    'flex_1': 19,
    'flex_2': 7,
    'flex_3': -5,
    'flex_4': -7,
    'flex_5': 1,
    'GYRx': -0.007634,
    'GYRy': -0.007634,
    'GYRz': -0.007634,
    'ACCx': 8.777173,
    'ACCy': 4.160693,
    'ACCz': 1.714282
}

# Convert hardcoded input data into a DataFrame to match the model's expected input
input_df = pd.DataFrame([input_data])

# Scale the data using the same scaler as during training
input_scaled = scaler.transform(input_df)

# Make a prediction
predicted_class = model.predict(input_scaled)

# Decode the prediction to get the gesture label
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'bad', 'deaf', 'fine', 'good', 'goodbye', 'hello', 'hungry', 'me', 'no', 'please']
predicted_gesture = alphabets[predicted_class[0]]

print(f"Predicted Gesture: {predicted_gesture}")


# In[9]:


import socket
import pandas as pd
from joblib import load
import pyttsx3
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

engine = pyttsx3.init()

# ESP32 IP address and port
HOST = ''
PORT = 4444

# Function to receive sensor data from ESP32
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            tmp=0
            while tmp<3:
                data = conn.recv(44)
                print("Connection received\n")
               
                if not data:
                    break
                    
                
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
       
               

                




                time.sleep(1)  # Adjust the delay as needed
                
                tmp=tmp+1

   


# In[ ]:





# In[ ]:




