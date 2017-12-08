
# coding: utf-8

# In[1]:

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout,Flatten, Dense
from keras import applications


# In[2]:

img_width, img_height = 150,150

train_data_dir = '../dataset/split_data/train/'
validation_data_dir = '../dataset/split_data/validation/'
nb_train_samples = 13447
nb_validation_samples = 4487
epochs = 3
batch_size = 16

bottleneck_features_train_file = 'bottleneck_features_train.npy'
bottleneck_features_validation_file = 'bottleneck_features_validation.npy'


# In[9]:

def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    
#     train_datagen = ImageDataGenerator(
#         rescale=1. / 255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

#     train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
    
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    
    print("Predicting TRAINING bottleneck features...")
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size, verbose=1)
    print("Saving TRAINING bottleneck features...")
    np.save(open(bottleneck_features_train_file, 'w'),
            bottleneck_features_train)
    print("SAVED in", bottleneck_features_train_file)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("Predicting VALIDATION bottleneck features...")
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size, verbose=1)
    print("Saving VALIDATION bottleneck features...")
    np.save(open(bottleneck_features_validation_file, 'w'),
            bottleneck_features_validation)
    print("SAVED in", bottleneck_features_validation)


# In[7]:

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))


# In[5]:

save_bottlebeck_features()


# In[ ]:



