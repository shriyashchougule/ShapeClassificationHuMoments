# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:33:00 2020

@author: SCHOUGU1
"""

from sklearn.externals import joblib
import sys
import cv2
import mahotas

args = sys.argv
model, scaler, labels = joblib.load(args[1])

test_dir = args[2]
print(test_dir)
test_img = os.listdir(test_dir)

prediction = open("Output.txt","w")

for img in test_img:
    img_pth = os.path.join(test_dir, img)
    print(img_pth)
    image = cv2.imread(img_pth)
    image = cv2.resize(image, (400,400))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f1 = cv2.HuMoments(cv2.moments(image_gray)).flatten()
    print(f1.shape)
    #print(f1)
    f2 = mahotas.features.haralick(image_gray).mean(axis=0)
    print(f2.shape)
    #print(f2)
    feature = np.hstack([f2, f1])
    feature = scaler.transform(feature.reshape(1,-1))
    predlabel = labels[model.predict(feature)[0]]
    print("prediction label: {}".format(predlabel))
    msg = "%s, %s \n" % (img, predlabel[-1])
    prediction.write(msg)
prediction.close()