from keras.preprocessing.image import ImageDataGenerator
from helper import target_size, batch_size, num_classes, create_top_model
import numpy as np
from keras.utils.np_utils import to_categorical

datagen = ImageDataGenerator(rescale=1.0/255.0) 

# ---------- GET TESTING DATA ----------

# create datagen and train generator to load the data from directory
test_generator = datagen.flow_from_directory(
                        '../../../dataset/split_data/test/', 
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode='categorical', # specify categorical
                        shuffle=False) # data is ordered

# load vgg features
test_data = np.load('vgg_test_features.npy')

test_labels = test_generator.classes
test_labels_onehot = to_categorical(test_labels, num_classes=num_classes)

# ---------- TEST MODEL ----------

model = create_top_model("softmax", test_data.shape[1:])
model.load_weights("_top_model_weights.h5")  

print(test_data.shape)
print(test_data.shape[1:])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

loss, acc = model.evaluate(test_data, test_labels_onehot, batch_size=batch_size, verbose=1)        

print()
print("loss: ", loss)
print("accuracy: {:8f}%".format(acc*100))