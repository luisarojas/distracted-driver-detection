
# coding: utf-8

# In[ ]:

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:

model = create_model()
model.load_weights("weights.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:

batch_size = 16
test_datagen = ImageDataGenerator(
            rotation_range=10, # range (0-180) within which to randomly rotate pictures
            # width_shift_range=0.2, # as fraction of width, range within to which randomly translate pictures
            # height_shift_range=0.2, # same as above, but with height
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            # shear_range=0.2, # random shearing transformations
            zoom_range=0.1, # randomly zooming inside pictures
            horizontal_flip=False,
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift

# this is a similar generator, for test data
test_generator = test_datagen.flow_from_directory(
        '../dataset/split_data/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:

score, acc = model.evaluate_generator(test_generator, 4490)

print("score: ", score)
print("accuracy:", acc)

