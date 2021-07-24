# Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
from itertools import chain
from random import sample   

# Load NIH data
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv') #provide path to where images saved
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.head(3)

# Analyse an image
all_xray_df['path']
img = imread('/data/images_012/images/00030801_001.png')
imshow(img)
img_mean = np.mean(img)
img_mean
img_std = np.std(img)
img_std

# Visualise data
plt.figure(figsize = (10,8))
plt.hist(all_xray_df['Patient Age'], bins=30)
plt.xlim([0,100])

plt.figure(figsize=(10,6))
all_xray_df['Patient Gender'].value_counts().plot(kind='bar')

plt.figure(figsize=(10,6))
all_xray_df['View Position']. value_counts().plot(kind='bar')

# Defining labels
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)

# Analyse pneumonia cases
len(all_xray_df[all_xray_df.Pneumonia==1])
len(all_xray_df[all_xray_df.Pneumonia==0])

plt.figure(figsize=(10,6))
plt.hist(all_xray_df[all_xray_df.Pneumonia==1]['Patient Age'], bins=30)
plt.xlim(0,100)

plt.figure(figsize=(10,6))
all_xray_df[all_xray_df.Pneumonia==1]['Patient Gender'].value_counts().plot(kind='bar')

# All labels
ax = all_xray_df[all_labels].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')

all_xray_df[all_labels].sum()/len(all_xray_df)

# All labels co-occurance with pneumonia
ax = all_xray_df[all_xray_df.Pneumonia==1][all_labels].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with pneumonia and co-occur diseases')

all_xray_df[all_xray_df.Pneumonia==1][all_labels].sum()/len(all_xray_df[all_xray_df.Pneumonia==1])

# Defining number of conditions per patient
all_xray_df['Disease Per Patient'] = all_xray_df[all_labels].sum(axis=1)
all_xray_df.sample(3)

all_xray_df['Disease Per Patient']. value_counts().plot(kind='bar')

# Read sample_labels.csv
sample_df = pd.read_csv('sample_labels.csv')
sample_df.sample(3)

sample_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(sample_paths), ', Total Headers', sample_df.shape[0])
sample_df['path'] = sample_df['Image Index'].map(sample_paths.get)
sample_df.head(3)

sample_labels = np.unique(list(chain(*sample_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
sample_labels = [x for x in sample_labels if len(x)>0]
print('Sample Labels ({}): {}'.format(len(sample_labels), sample_labels))
for c_label in sample_labels:
    if len(c_label)>1: # leave out empty labels
        sample_df[c_label] = sample_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
sample_df.sample(3)

# Pixel-wise analysis
pneumonia = sample_df[sample_df.Pneumonia==1]
pneumonia['path']
pneumonia = imread('/data/images_012/images/00028924_005.png')
imshow(pneumonia)

normal_x = [700,750]
normal_y = [300,400]
pneumonia_x = [200,250]
pneumonia_y = [600,700]

plt.figure(figsize=(5,5))
plt.hist(pneumonia[normal_y[0]:normal_y[1],normal_x[0]:normal_x[1]].ravel(), bins = 256, color='blue')
plt.hist(pneumonia[pneumonia_y[0]:pneumonia_y[1], pneumonia_x[0]:pneumonia_x[1]].ravel(), bins = 256, color='red')
plt.legend(['Normal', 'Pneumonia'])
plt.xlabel('intensity')
plt.ylabel('number of pixels')
plt.show()

# Different approach 
cardiomegaly = sample_df[sample_df.Cardiomegaly==1]
cardiomegaly['path']

cardiomegaly = imread('/data/images_012/images/00030419_001.png')
plt.imshow(cardiomegaly)
plt.hist(cardiomegaly.ravel(), bins=256)

plt.imshow(pneumonia)
plt.hist(pneumonia.ravel(), bins = 256)

thresh = 100
cardiomegaly_bin = (cardiomegaly>thresh) * 255
pneumonia_bin = (pneumonia>thresh) * 255
plt.imshow(cardiomegaly_bin)

plt.imshow(pneumonia_bin)

cardiomegaly = sample_df[sample_df.Cardiomegaly==1]
pneumonia = sample_df[sample_df.Pneumonia==1]

cardiomegaly_intensities = []
for i in cardiomegaly['path']:
    img = imread(i)
    img_mask = img>thresh
    cardiomegaly_intensities.extend(img[img_mask].tolist())
plt.hist(cardiomegaly_intensities, bins =256)

pneumonia_intensities = []
for i in pneumonia['path']:
    img = imread(i)
    img_mask = img>thresh
    pneumonia_intensities.extend(img[img_mask].tolist())
plt.hist(pneumonia_intensities, bins =256)





