### ---------- Handle Command Line Arguments ----------

import argparse

a = argparse.ArgumentParser(description="Extracting the deep features from a pre-trained VGG16 model, using our own training data.")
args = a.parse_args()

### ---------- Import Relevand Libraries ----------

from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from helper import target_size, batch_size
from math import ceil
import numpy as np

datagen = ImageDataGenerator(rescale=1./255)

# load vgg16 model, excluding the top fully connected layers
model = applications.VGG16(include_top=False, weights='imagenet')

# ---------- TRAINING DATA----------

# run training images through vgg and obtain its deep features (until last convolutional layer)
train_generator = datagen.flow_from_directory(
                    '../../../dataset/split_data/train/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_train_samples = len(train_generator.filenames)

# obtain steps required per epoch
train_steps = ceil(num_train_samples/batch_size)

# obtain deep/bottleneck features from vgg for the training data and save them
vgg_train_features = model.predict_generator(train_generator, steps=train_steps, verbose=1)
print('Saving deep features for training data...')
np.save('res/vgg_train_features.npy', vgg_train_features)

# ---------- VALIDATION DATA----------

# run validation images through vgg and obtain its deep features (until last convolutional layer)
val_generator = datagen.flow_from_directory(
                    '../../../dataset/split_data/validation/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_val_samples = len(val_generator.filenames)

# obtain steps required per epoch
val_steps = ceil(num_val_samples/batch_size)

# obtain deep/bottleneck features from vgg for the validation data and save them
vgg_val_features = model.predict_generator(val_generator, steps=val_steps, verbose=1)
print('Saving deep features for validation data...')
np.save('res/vgg_val_features.npy', vgg_val_features)

# ---------- TESTING DATA----------

# run testing images through vgg and obtain its deep features (until last convolutional layer)
test_generator = datagen.flow_from_directory(
                    '../../../dataset/split_data/test/',
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=None, # data without labels
                    shuffle=False)

num_test_samples = len(test_generator.filenames)

# obtain steps required per epoch
test_steps = ceil(num_test_samples/batch_size)

# obtain deep/bottleneck features from vgg for the testing data and save them
vgg_test_features = model.predict_generator(test_generator, steps=test_steps, verbose=1)
print('Saving deep features for testing data...')
np.save('res/vgg_test_features.npy', vgg_test_features)