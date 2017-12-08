
# coding: utf-8

# ## Using the bottleneck features of a pre-trained network

# "A pretrained network would have already learned features that are useful for most computer vision problems, and leveraging such features would allow us to reach a better accuracy than any method that would only rely on the available data."
# 
# We will use the VGG16 architecture.
# 
# "We will only instantiate the convolutional part of the model, everything up to the fully-connected layers. We will then run this model on our training and validation data once, recording the output (the "bottleneck features")."
# 
# "The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base and running the whole thing, is computational effiency. Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. Note that this prevents us from using data augmentation."
# 
# In short, we will:
# 1. Save the bottleneck features from the VGG16 model.
# 2. Train a small network using the saved features to classify our classes, and save that model (the "top model")
# 3. Use both the VGG16 model and the top model to make predictions.

# In[ ]:

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt  
import math


# In[ ]:

target_size=(224, 224)
path_top_model_weights = "bottleneck_fc_model.h5"
epochs = 5 # number of epochs to train the top model
batch_size = 16 # to be used by flow_from_directory and predict_generator


# In[ ]:

# save the bottleneck features from the VGG16 model
def save_bottleneck_features():
    
    # create the VGG16 model without the final fully-connected layers and load the ImageNet weights
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    # create the data generator for training images
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
                        '../dataset/split_data/train/',
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=None, # generator will only yield batches of data, no labels
                        shuffle=False) # the data will be in order
    
    # run the training images on the VGG16 model to save the bottleneck features
    num_train_samples = len(train_generator.filenames)
    num_classes = len(train_generator.class_indices)
    
    size_train_prediction = int(math.ceil(num_train_samples/batch_size)) # calculate the number of iterations when working
                                                                      # on batches when the number of training samples
                                                                      # isn't divisible by the batch size
            
    bottleneck_features_train = model.predict_generator(train_generator, size_train_prediction)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    # now, do the same with the validation images
    val_generator = datagen.flow_from_directory(
                        '../dataset/split_data/validation/',
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode=None, # generator will only yield batches of data, no labels
                        shuffle=False) # the data will be in order
    
    num_val_samples = len(val_generator.filenames)
    
    size_val_prediction = int(math.ceil(num_val_samples/batch_size)) # calculate the number of iterations when working
                                                                      # on batches when the number of training samples
                                                                      # isn't divisible by the batch size
            
    bottleneck_features_val = model.predict_generator(val_generator, size_val_prediction)
    
    np.save('bottleneck_features_val.npy', bottleneck_features_val)


# In[ ]:

# train the top model
def train_top_model():
    
    # ---------- PREPARE DATA FOR TRAINING ----------
    
    top_datagen = ImageDataGenerator(rescale=1./255)
    
    # to train the top model, we need class labels for each of the samples in training and validation
    top_model_train_generator = top_datagen.flow_from_directory(
                            '../dataset/split_data/train/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False)
    
    num_train_samples = len(top_model_train_generator.filenames)
    num_classes = len(top_model_train_generator.class_indices)
    
    # load the previously saved bottleneck features
    vgg_train_data = np.load('bottleneck_features_train.npy')
    
    # get class labels and convert them to categorical vectors
    train_labels = to_categorical(top_model_train_generator.classes, num_classes=num_classes)
    
    # repeat the process with validation images
    top_model_val_generator = top_datagen.flow_from_directory(
                            '../dataset/split_data/validation/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False)
    
    num_val_samples = len(top_model_val_generator.filenames)
    
    # load the previously saved bottleneck features
    vgg_val_data = np.load('bottleneck_features_val.npy')
    
    # get class labels and convert them to categorical vectors
    val_labels = to_categorical(top_model_val_generator.classes, num_classes=num_classes)
    
    # ---------- CREATE AND TRAIN THE TOP MODEL ----------
    # use the bottleneck features as input
    
    model = Sequential()
    model.add(Flatten(input_shape=vgg_train_data.shape[1:]))  
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
                vgg_train_data,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(vgg_val_data, val_labels))
    
    model.save_weights(path_top_model_weights)
    
    # ---------- TEST MODEL ----------
    
    loss, acc = model.evaluate(  
                        vgg_val_data,
                        val_labels,
                        batch_size=batch_size,
                        verbose=1)

    print("loss: {}".format(loss))
    print("accuracy: {:.5f}%".format(acc * 100))
    
    plt.figure(1)  
    
    # ---------- PLOT TRAINING AND TESTING RESULTS ----------
   
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[ ]:

save_bottleneck_features()
train_top_model()


# In[ ]:



