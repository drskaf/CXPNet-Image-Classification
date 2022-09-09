# Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
from skimage.io import imread, imshow
from itertools import chain
from random import sample    
from keras.preprocessing.image import ImageDataGenerator

# Load data   
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# Create labels
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(50)

# Create a anew variable 'pneumonia class'
conditions = [all_xray_df.Pneumonia==1, all_xray_df.Pneumonia==0]
values = ['pneumonia', 'no_pneumonia']
all_xray_df['pneumonia_class'] = np.select(conditions, values)

# Create training/testing split
train_df, valid_df = skl.train_test_split(all_xray_df, test_size=0.2, stratify=all_xray_df['Pneumonia'])
train_df['Pneumonia'].sum()/len(train_df)

valid_df['Pneumonia'].sum()/len(valid_df)

p_inds = train_df[train_df.Pneumonia==1].index.tolist()
np_inds = train_df[train_df.Pneumonia==0].index.tolist()
np_sample = sample(np_inds,len(p_inds))
train_df = train_df.loc[p_inds + np_sample]
train_df['Pneumonia'].sum()/len(train_df)

p_inds = valid_df[valid_df.Pneumonia==1].index.tolist()
np_inds = valid_df[valid_df.Pneumonia==0].index.tolist()
np_sample = sample(np_inds,4*len(p_inds))
valid_df = valid_df.loc[p_inds + np_sample]
valid_df['Pneumonia'].sum()/len(valid_df)

len(train_df)
len(valid_df)

# BUILD MODEL

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50 
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Import VGG16 and freeze 17 layers     
model = VGG16(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False
    
for layer in vgg_model.layers:
    print(layer.name, layer.trainable)
    
# Fine tuning
my_model = Sequential()  
my_model.add(vgg_model)
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(1, activation='sigmoid'))
    
optimizer = Adam(lr=1e-4)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']

# Data augmentation
train_idg = ImageDataGenerator(samplewise_center=True,
                               samplewise_std_normalization=True,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=10, 
                              shear_range = 0.1,
                              zoom_range=0.1)



train_gen = train_idg.flow_from_dataframe(dataframe=train_df, directory=None, 
                                          x_col = 'path',
                                          y_col = 'pneumonia_class',
                                          class_mode = 'binary',
                                          target_size = (224,224), 
                                          batch_size = 64
                                         )

    
val_idg = ImageDataGenerator(samplewise_center=True,
                               samplewise_std_normalization=True,
                                 )


val_gen = val_idg.flow_from_dataframe(dataframe=valid_df, 
                                      directory=None, 
                                      x_col = 'path',
                                      y_col = 'pneumonia_class',
                                      class_mode = 'binary',
                                      target_size = (224,224), 
                                      batch_size = 32)
valX, valY = val_gen.next()

# Model checkpoints
weight_path="{}_my_model.best.hdf5".format('#CPXNet')

checkpoint = ModelCheckpoint(weight_path, 
                             monitor= 'val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode= 'min', 
                             save_weights_only = False)

early = EarlyStopping(monitor= 'val_loss', 
                      mode= 'min', 
                      patience=10)
  
callbacks_list = [checkpoint, early]

# Train model
my_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = my_model.fit_generator(train_gen, 
                          validation_data = (valX, valY), 
                          epochs = 50, 
                          callbacks = callbacks_list)

# Evaluate model. NB: run evaluate_model.py script first to load the functions
plot_history(history)

my_model_4.load_weights(weight_path)
pred_Y = my_model_4.predict(valX, batch_size = 32, verbose = True)

plot_auc(valY, pred_Y)

plot_precision_recall_curve(valY, pred_Y)

# Create plot to show F1 Score vs Threshold
from sklearn.metrics import f1_score
from sklearn.preprocessing import binarize
from numpy import argmax
precision, recall, thresholds = precision_recall_curve(valY.astype(int), pred_Y)
f1_scores = []
for i in thresholds:
    f1 = f1_score(valY.astype(int), binarize(pred_Y,i))
    f1_scores.append(f1)
    
plt.plot(f1_scores,thresholds)
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))

# Look at some examples
fig, m_axs = plt.subplots(2, 2, figsize = (16, 16))
i = 0
for (c_x, c_y, c_ax) in zip(valX[0:100], valY[0:100], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        if pred_Y[i] > 0.47:  
            c_ax.set_title('1, 1')
        else:
            c_ax.set_title('1, 0')
    else:
        if pred_Y[i] > 0.47: 
            c_ax.set_title('0, 1')
        else:
            c_ax.set_title('0, 0')
    c_ax.axis('off')
    i=i+1
         
# Just save model architecture to a .json:

model_json = my_model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)   
