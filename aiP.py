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













 

