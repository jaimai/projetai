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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns

#
# PARAMETERS
#

dataFolder = './img_align_celeba'
trainingFolder = './Training'
testingFolder = './Testing'

csvFile = './list_attr_celeba.csv'
argument_idx = 21   # Male or female arg

saveModelFile = './save.h5'
batch_size = 50
img_size = (64, 64)
input_shape = (64, 64, 3)
epochs = 5

#
# Part 1 : Preprocessing
#

# import datset
dataset = pd.read_csv(csvFile)

# recover data
img_id_list = dataset.iloc[:, 0].values
argument_list = dataset.iloc[:, argument_idx].values

# replace -1 into 0 in argument_list
argument_list = np.where(argument_list == -1, 0, argument_list)

# remove folders
shutil.rmtree(trainingFolder, ignore_errors=True)
shutil.rmtree(testingFolder, ignore_errors=True)

# Creating folders
os.makedirs(os.path.join(trainingFolder, 'Presence_of_feature'), exist_ok=True)
os.makedirs(os.path.join(trainingFolder, 'Absence_of_feature'), exist_ok=True)
os.makedirs(os.path.join(testingFolder, 'Presence_of_feature'), exist_ok=True)
os.makedirs(os.path.join(testingFolder, 'Absence_of_feature'), exist_ok=True)

# Split dataset into training test set
x_train, x_test, y_train, y_test = train_test_split(img_id_list, argument_list, test_size=0.2)


# Copy img to folder fct
def img_to_folder(name_folder, img_list, arg_list):
    for i in range(len(img_list)):
        src = os.path.join(dataFolder, img_list[i])
        if arg_list[i] == 1:
            dst = os.path.join(name_folder, 'Presence_of_feature', img_list[i])
        else:
            dst = os.path.join(name_folder, 'Absence_of_feature', img_list[i])
        shutil.copyfile(src, dst)


img_to_folder(trainingFolder, x_train, y_train)
img_to_folder(testingFolder, x_test, y_test)

#
# Part 2 : building model
#

# init CNN 
classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Flatten())

# init ANN
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#
# Part 3 : Training the model using training data
#

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

training_set = train_datagen.flow_from_directory(trainingFolder,
                                                 target_size=img_size,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 subset='training',
                                                 class_mode='binary')

validation_set = train_datagen.flow_from_directory(trainingFolder,
                                                   target_size=img_size,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   subset='validation',
                                                   class_mode='binary')

hist = classifier.fit_generator(
    training_set,
    steps_per_epoch=(training_set.n // batch_size) + 1,
    epochs=epochs,
    validation_data=validation_set,
    validation_steps=(validation_set.n // batch_size) + 1,
    verbose=1
)

# save the model
classifier.save(saveModelFile)

#
# Part 4 : Prediction using testing data
#

# Dataset test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(testingFolder,
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            class_mode='binary',
                                            shuffle=False)

# Evaluate model
loss_v, acc_v = classifier.evaluate_generator(test_set, steps=(test_set.n // batch_size) + 1, verbose=1)
test_set.reset()

# Predict images
y_predict = classifier.predict_generator(test_set, steps=(test_set.n // batch_size) + 1, verbose=1)
y_test = test_set.classes[test_set.index_array]

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict > 0.5)
cm = pd.DataFrame(data=cm, index=["Male", "Female"], columns=["Male", "Female"])

#
# Part 5 : Visualisation
#

# Visualising confusion matrix using a heatmap
plt.figure(figsize=(9, 9))
ax = sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues_r')
ax.set_ylim([2, 0])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
title = 'Accuracy Score: {0:.5f}'.format(acc_v)
plt.title(title, size=15)
plt.show()

# Visualising loss evolution during training
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Visualising accuracy evolution during training
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
