from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

img_path = "../../../dataset/split_data/test/c5/img_79836.jpg"

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

predicted = model.predict_classes(bottleneck_features)
predicted_onehot = to_categorical(predicted, num_classes=num_classes)

print(predicted_onehot)