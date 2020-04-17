#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:17:48 2020

@author: Thomas / Mathias / Pierre / Felix 
"""
import pandas as pd
import numpy as np
import os 
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image


#
# Part 1 : preprocessing
#

#import datset
dataset = pd.read_csv('list_attr_celeba.csv')

#recover data 
img_id_list = dataset.iloc[:, 0].values
attractive_list = dataset.iloc[:, 3].values

#replace -1 into 0 in attractive_list
attractive_list = np.where(attractive_list == -1, 0,attractive_list)

#remove folders
shutil.rmtree('./Testing', ignore_errors=True)
shutil.rmtree('./Trainning', ignore_errors=True)


#Creating folders 
os.makedirs('./Trainning/Presence_of_feature', exist_ok=True)
os.makedirs('./Trainning/Absence_of_feature', exist_ok=True)
os.makedirs('./Testing/Presence_of_feature', exist_ok=True)
os.makedirs('./Testing/Absence_of_feature', exist_ok=True)

#Split dataset into trainning test set
x_train, x_test, y_train, y_test = train_test_split(img_id_list, attractive_list, test_size=0.2)

#Copy img to folder

def img_to_folder(name_folder, img_list, attractive_list): 
    for i in range(len(img_list)):
        src=os.path.join('img_align_celeba/',img_list[i])
        if attractive_list[i]  == 1:
            dst = os.path.join(name_folder, 'Presence_of_feature', img_list[i])
        else :
            dst = os.path.join(name_folder, 'Absence_of_feature', img_list[i])  
        shutil.copyfile(src, dst)
        
img_to_folder('./Trainning',x_train,y_train)
img_to_folder('./Testing',x_test,y_test)


#
# Part 2 : building model
#

# init CNN 
classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Flatten())

# init ANN
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#
# Part 3 : fitting images to CNN
#

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

training_set = train_datagen.flow_from_directory('./Trainning',
                                                 target_size=(64,64), 
                                                 batch_size=32, 
                                                 class_mode='binary')

hist = classifier.fit_generator(
        training_set, 
        steps_per_epoch=(len(x_train)/32),
        epochs=5
        )

# save the model
classifier.save('attrative_cnn.h5')

#
# Part 4 : Prediction 
#

# Predict image
#test_image = image.load_img('./image_test/img3.jpg', target_size=(64,64))
#test_image = image.img_to_array(test_image).astype('float32')/255
#test_image = np.expand_dims(test_image, axis=0)

#result = classifier.predict(test_image)

# Predict images
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = train_datagen.flow_from_directory('./Testing',
                                                 target_size=(64,64), 
                                                 batch_size=32, 
                                                 class_mode='binary')
y_predict = classifier.predict_generator(test_set,steps=len(test_set))

# Confusion Matrix
y_predict = y_predict > 0.5
cm = confusion_matrix(y_test, y_predict)













 

