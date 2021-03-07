# Import packages
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import keras 
from keras.models import load_model
import json

# Check headers of test DICOM images
test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']
dcm = pydicom.dcmread(test_dicoms[0])
dcm

# Define functions
def check_dicom(filename): 
    
    
    print('Load file {} ...'.format(filename))
    ds = pydicom.dcmread(filename) 
    img = ds.pixel_array
    if ds.Modality=='DX' and ds.BodyPartExamined=='CHEST'and ds.PatientPosition=='PA':
        return img  
    elif ds.Modality=='DX' and ds.BodyPartExamined=='CHEST'and ds.PatientPosition=='AP':
        return img
    else:
        print('Invalid DICOM')
        return None
      
def preprocess_image(img,img_mean, img_std,img_size): 
    # todo
    img_mean = np.mean(img)
    img_std = np.std(img)
    new_img = img.copy()
    new_img = (new_img - img_mean/img_std)
    img_proc = np.resize(new_img, img_size)
    return img_proc
  
def predict_image(model, img_proc, thresh): 
    # todo    
    model.load_weights(weight_path)
    pred = model.predict(img_proc)
    if pred > thresh:
        return 'Pneumonia'
    else:
        return 'No Pneumonia'
      
      
test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path = ('/home/workspace/my_model.json')
weight_path = ('/home/workspace/#11_experiment_my_model.best.hdf5')

IMG_SIZE=(1,224,224,3)

my_model = load_model(weight_path) #loads model
thresh = 0.52  #Use the threshold from your best model

for i in test_dicoms:
    
    img = np.array([])
    img = check_dicom(i)
    
    if img is None:
        continue
    
    img_mean = np.mean(img)
    img_std = np.std(img)   
           
    img_proc = preprocess_image(img,img_mean, img_std,IMG_SIZE)
    pred = predict_image(my_model_4, img_proc, thresh)
    print(pred)
