### ---------- Handle Command Line Arguments ----------

import argparse

a = argparse.ArgumentParser(description="Generate the missing top layers for the VGG16 model, then train it on own training data.")
args = a.parse_args()

### ---------- Import Relevant Libraries ----------

from helper import create_top_model, num_classes, class_labels, target_size, batch_size
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# global variables
epochs = 50
datagen = ImageDataGenerator(rescale=1./225)

# ---------- LOAD TRAINING DATA ----------

# create datagen and train generator to load the data from directory
train_generator = datagen.flow_from_directory(
                            '../../../dataset/split_data/train/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False) # data is ordered
                            
num_train_samples = len(train_generator.filenames)

# load vgg features
train_data = np.load('res/vgg_train_features.npy')

train_labels = train_generator.classes
train_labels_onehot = to_categorical(train_labels, num_classes=num_classes)

# ---------- LOAD VALIDATION DATA ----------

# create datagen and train generator to load the data from directory
val_generator = datagen.flow_from_directory(
                            '../../../dataset/split_data/validation/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False) # data is ordered
                            
num_val_samples = len(val_generator.filenames)

# load vgg features
val_data = np.load('res/vgg_val_features.npy')

val_labels = val_generator.classes
val_labels_onehot = to_categorical(val_labels, num_classes=num_classes)

# ---------- CREATE AND TRAIN MODEL ----------

# create the top model to be trained
model = create_top_model("softmax", train_data.shape[1:])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# only save the best weights. if the accuracy doesnt improve in 2 epochs, stop.
checkpoint_callback = ModelCheckpoint(
                        "res/top_model_weights.h5", # store weights with this file name
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stop_callback = EarlyStopping(
                        monitor='val_acc',
                        patience=2, # max number of epochs to wait
                        mode='max') 

callbacks_list = [checkpoint_callback, early_stop_callback]

# train the model
history = model.fit(
            train_data,
            train_labels_onehot,
            epochs=epochs,
            batch_size=batch_size,
            # validation_data=val_data,
            validation_data=(val_data, val_labels_onehot),
            callbacks=callbacks_list)