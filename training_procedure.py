# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:39:05 2020

@author: SCHOUGU1
"""

import numpy as np
import mahotas
import cv2
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

training_images_path = r"C:\Users\schougu1\pythonscripts\computer vision"
class_dir = os.listdir(training_images_path)
print(class_dir)

all_features = []
#labels = np.array([1,2,3,4])
img_labels = []

for dir_name in class_dir:

    dir_class_images = os.path.join(training_images_path, dir_name)

    for img_name in os.listdir(dir_class_images):
        file = os.path.join(dir_class_images, img_name)
        print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (400, 400))
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f1 = cv2.HuMoments(cv2.moments(image_gray)).flatten()
        print(f1.shape)
        #print(f1)
        f2 = mahotas.features.haralick(image_gray).mean(axis=0)
        print(f2.shape)
        #print(f2)
        feature = np.hstack([f2, f1])
        print(feature.shape)
        all_features.append(feature)
        img_labels.append(dir_name)
        
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_features = scaler.fit_transform(all_features)
print(normalized_features)
print(img_labels)
labels = np.unique(img_labels)
encode = LabelEncoder()
target_labels = encode.fit_transform(img_labels)
print(target_labels)

classifiers = []

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier

rand_seed = 9 
classifiers.append(('LDA', LinearDiscriminantAnalysis()))
classifiers.append(('SGD', SGDClassifier(max_iter=1000, tol=1e-3)))
classifiers.append(('LR', LogisticRegression(random_state=rand_seed)))
classifiers.append(('KNC', KNeighborsClassifier()))
classifiers.append(('SVM', SVC()))
#classifiers.append(('DTC', DecisionTreeClassifier()))
classifiers.append(('NB', GaussianNB()))

global_features = np.array(normalized_features)
global_labels   = np.array(target_labels)

from sklearn.model_selection import train_test_split, cross_val_score
(train_img, test_img, train_label, test_label) = train_test_split(np.array(global_features),
                                                    np.array(global_labels),test_size=0.1, random_state=rand_seed)

        

from sklearn.model_selection import KFold

cv_results = []
mean_accuracy = []
names = []
for name, model in classifiers:
    kfold = KFold(n_splits=10, random_state=rand_seed)
    cv_results = cross_val_score(model, train_img, train_label, cv=kfold, scoring="accuracy")
    #results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    mean_accuracy.append(cv_results.mean())
    print(msg)
    
name, selected_model = classifiers[mean_accuracy.index(max(mean_accuracy))]
print("Model Selected {}".format(name))
selected_model.fit(global_features, global_labels)

joblib.dump([selected_model, scaler, labels], 'model_data.sav')