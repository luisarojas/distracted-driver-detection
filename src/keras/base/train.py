### ---------- Handle Command Line Arguments ----------

import argparse

a = argparse.ArgumentParser(description="Train a simple CNN model and save improved weights.")
a.add_argument("--bsize", help="provide batch size for training (default: 40)", type=int, default=40)
args = a.parse_args()

### ---------- Import Relevand Libraries ----------

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from model import create_model
from math import ceil

model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Prepare the data

# using flow_from_directory(), generate batches of image data (and their labels)
batch_size = args.bsize

train_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            # width_shift_range=0.2, # as fraction of width, range within to which randomly translate pictures
            # height_shift_range=0.2, # same as above, but with height
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            # shear_range=0.2, # random shearing transformations
            zoom_range=0.1, # randomly zooming inside pictures
            horizontal_flip=False,
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift

val_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            # width_shift_range=0.2, # as fraction of width, range within to which randomly translate pictures
            # height_shift_range=0.2, # same as above, but with height
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            # shear_range=0.2, # random shearing transformations
            zoom_range=0.1, # randomly zooming inside pictures
            horizontal_flip=False,
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift


# generate batches of augmented image data, given the path of the original images

# instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory)
# this generator will read pictures found in subfolders and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../../../dataset/split_data/train/',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
val_generator = val_datagen.flow_from_directory(
        '../../../dataset/split_data/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


# use the generators above to train the model
# stop when the val accuracy stops getting better

filepath="weights.h5"

checkpoint_callback = ModelCheckpoint(
                        filepath,
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stop_callback = EarlyStopping(
                monitor='val_acc',
                # verbose=0, # decides what to print
                # min_delta=0, # threshold to whether quantify a loss at some epoch as improvement or not. If the difference of loss is below min_delta, it is quantified as no improvement
                # mode='auto', # depends on the direction of the monitored quantity (is it supposed to be decreasing or increasing), since we monitor the loss, we can use min.        
                patience=3,
                mode='max') 

callbacks_list = [checkpoint_callback, early_stop_callback]

history = model.fit_generator(
            train_generator,
            steps_per_epoch=(ceil(len(train_generator.filenames) // batch_size)),
            epochs=50,
            validation_data=val_generator,
            validation_steps=(ceil(len(val_generator.filenames) // batch_size)),
            callbacks=callbacks_list)



