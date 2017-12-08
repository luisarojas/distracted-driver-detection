from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from model import create_model
import numpy as np
import operator
import matplotlib.pyplot as plt

img_path = "../../../dataset/split_data/test/c0/img_41813.jpg"
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']

model = create_model()
model.load_weights("_weights.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

target_size=(150,150)

# prepare image for classification using keras utility functions
image = load_img(img_path, target_size=target_size)

image_arr = img_to_array(image) # convert from PIL Image to NumPy array
# the dimensions of image should now be (150, 150, 3)

# to be able to pass it through the network and use batches, we want it with shape (1, 224, 224, 3)
image_arr = np.expand_dims(image_arr, axis=0)

data_datagen = ImageDataGenerator(
            rescale=1./255, # RBG coefficient values 0-255 are too hight to process. instead, represent them as values 0-1
            fill_mode='nearest') # strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift

# this is a similar generator, for test data
data_generator = data_datagen.flow(image_arr)

# classify given an image
predictions = model.predict_generator(data_generator)

# get human-readable labels of the preditions, as well as the corresponding probability
decoded_predictions = dict(zip(class_labels, predictions[0]))

# sort dictionary by value
decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

count = 1
for key, value in decoded_predictions[:5]:
    print("{}. {}: {:8f}%".format(count, key, value*100))
    count+=1

# print image
plt.imshow(image)
plt.axis('off')
plt.show()