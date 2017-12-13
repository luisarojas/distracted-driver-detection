### ---------- Handle Command Line Arguments ----------

import argparse

a = argparse.ArgumentParser(description="Generate predictions for test images provided by Kaggle.")
a.add_argument("-f", "--filename", help='csv file name to save the generated predictions to (default: predictions.csv)', default='predictions.csv')
args = a.parse_args()

filename = args.filename

### ---------- Import Relevand Libraries ----------

from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size, num_classes
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import os

# iterate through all testing images
data_path = '../../../dataset/original/test/'
csv_header = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
all_entries = []

def predict_class(img_path):
    
    # prepare image for classification using keras utility functions
    image = load_img(img_path, target_size=target_size)

    image_arr = img_to_array(image) # convert from PIL Image to NumPy array
    image_arr /= 255

    # to be able to pass it through the network and use batches, we want it with shape (1, 224, 224, 3)
    image_arr = np.expand_dims(image_arr, axis=0)
    # print(image.shape)

    # build the VGG16 network  
    model = applications.VGG16(include_top=False, weights='imagenet')  

    # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_features = model.predict(image_arr) 

    # build top model  
    model = create_top_model("softmax", bottleneck_features.shape[1:])

    model.load_weights("res/_top_model_weights.h5")

    predicted = model.predict(bottleneck_features)
    # predicted_onehot = to_categorical(predicted, num_classes=num_classes)

    return np.asarray(predicted[0]) # float32

for i, file in enumerate(os.listdir(data_path)):
    file_name = os.fsdecode(file)

    if file_name.endswith(".jpg"):
        
        print(i, " ", file_name, "...")
        img_path = (os.path.join(data_path, file_name))
        
        predicted = predict_class(img_path)
        predicted = np.asarray(['%.1f'%num for num in predicted]).astype('str')
        
        entry = np.concatenate((np.array([file_name]), predicted))
        # entry = np.array2string(entry, separator=",").replace('\n', '')
        # entry = ','.join([num for num in entry])

        # print(str(entry))
        all_entries.append(entry)
        
all_entries = (np.asarray(all_entries))
all_entries = all_entries[np.argsort(all_entries[:,0])]
# print(all_entries)

import csv
with open(filename, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(csv_header)
    for row in all_entries:
        filewriter.writerow(row)