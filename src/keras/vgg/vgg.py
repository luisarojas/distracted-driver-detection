from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dropout,Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import operator
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from keras.callbacks import EarlyStopping, ModelCheckpoint


# parameters
epochs = 5
batch_size = 16

# constants
target_img_width, target_img_height = 224, 224
train_dir = '../../../dataset/split_data/train/'
val_dir = '../../../dataset/split_data/validation/'
test_dir = '../../../dataset/split_data/test/'
vgg_train_features_file = "vgg_train_features.npy"
vgg_val_features_file = "vgg_val_features.npy"
vgg_test_features_file = "vgg_test_features.npy"
num_classes = 10
top_model_weights_path = '_top_model_weights.h5'


# extract feature-vectors from VGG16
def get_features(model, data_dir):
    #Create a generator to load the data
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    generator = datagen.flow_from_directory(data_dir, 
                                            target_size=(target_img_width, target_img_height),
                                            batch_size=batch_size, 
                                            class_mode=None, # only the data, without labels
                                            shuffle=False) # keep data ordered 
    # extract information about the data
    num_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    
    # obtain number of steps required
    steps = ceil(num_samples / batch_size)
    #print("steps %s" % steps)
    
    # obtain the bottleneck features before the dense layers
    features = model.predict_generator(generator, steps=steps, verbose=1)
    return features

def extract_vgg16_features():
    
    # load the VGG16 Model
    model = applications.VGG16(include_top=False, weights="imagenet")
    
    #-----------------TRAINING DATA------------------
    # Run the training data through vgg and obtain the corresponding features
    train_features = get_features(model, train_dir)                
    
    # Save the training features in a numpy file
    np.save(vgg_train_features_file, train_features)
    print("Saved Training Features in %s" % vgg_train_features_file)
    
    #-----------------VALIDATION DATA------------------
    # Run the validation data through vgg and obtain the corresponding features
    val_features = get_features(model, val_dir)                
    
    #Save the validation features in a numpy file
    np.save(vgg_val_features_file, val_features)
    print("Saved Validation Features in %s" % vgg_val_features_file)
    
    #-----------------TESTING DATA------------------
    # Run the testing data through vgg and obtain the corresponding features
    test_features = get_features(model, test_dir)                
    
    # Save the testing features in a numpy file
    np.save(vgg_test_features_file, test_features)
    print("Saved Testing Features in %s" % vgg_test_features_file)
    

# # Top Model to be Retrained

# In[5]:

def create_top_model(final_activation,input_shape):
    
    model = Sequential()  
    model.add(Flatten(input_shape=input_shape))  
    model.add(Dense(256, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(num_classes, activation=final_activation)) # sigmoid to train, softmax for prediction
    
    return model


# In[6]:

def load_data_and_labels(features_file, data_dir):
    # Create the datagen
    datagen = ImageDataGenerator(rescale=1.0/255.0) 
        
    # Create the generator to load the data
    generator = datagen.flow_from_directory(data_dir, 
                                            target_size=(target_img_width, target_img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical', # specify categorical
                                            shuffle=False) # Data is ordered
    # Obtain information about the data
    num_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    
    # Load the training data features
    data = np.load(features_file)
    
    # Obtain class labels from the generator
    labels = generator.classes    
    # Convert into onehot 
    labels_onehot = to_categorical(labels, num_classes=num_classes)
    
    return data, labels_onehot


# In[7]:

def train_top_model():
    
    # Load the TRAINING data and labels
    train_data, train_labels = load_data_and_labels(vgg_train_features_file, train_dir)    
    
    # Load the VALIDATION data and labels
    val_data, val_labels = load_data_and_labels(vgg_val_features_file, val_dir)    
    
    # Create the top model to be trained
    model = create_top_model("sigmoid", train_data.shape[1:])
    
    # Compile the model
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint_callback = ModelCheckpoint(
                            "top_model_weights.h5",
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

    early_stop_callback = EarlyStopping(
                            monitor='val_acc',
                            patience=3,
                            mode='max') 

    callbacks_list = [checkpoint_callback, early_stop_callback]
    
    # Train the model
    history = model.fit(
                train_data,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_data, val_labels),
                callbacks=callbacks_list)
    
    # Save the trained weights of the model
    # model.save_weights(top_model_weights_path)    

    return model, history


# # Test the model

# In[8]:

def test_model(model):
    
    # Load the TESTING data and labels
    test_data, test_labels = load_data_and_labels(vgg_test_features_file, test_dir) 
    
    # Obtain a final Accuracy
    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)        
    
    print("------------TOTAL-----------")
    print("Final Accuracy =", accuracy*100, "%")
    print("Final Loss=", loss)


# # Prediction of single image

# In[9]:

def get_prediction_from_image(img_path):
    
    class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']

    target_size=(224,224)

    # prepare image for classification using keras utility functions
    image = load_img(img_path, target_size=target_size)
    
    # print image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    image = img_to_array(image) # convert from PIL Image to NumPy array
    image /= 255
    # the dimensions of image should now be (150, 150, 3)

    # to be able to pass it through the network and use batches, we want it with shape (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    # print(image.shape)

    # build the VGG16 network  
    model = applications.VGG16(include_top=False, weights='imagenet')  

    # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_prediction = model.predict(image) 
    
    # build top model  
    model = create_top_model("softmax", bottleneck_prediction.shape[1:])

    model.load_weights(top_model_weights_path)  

    # use the bottleneck prediction on the top model to get the final classification  
    class_predicted = model.predict_classes(bottleneck_prediction) 
    
    probs = model.predict(bottleneck_prediction) 
    decoded_predictions = dict(zip(class_labels, probs[0]))
    decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)
    
    count = 1
    for key, value in decoded_predictions[:5]:
        print("{}. {}: {:8f}%".format(count, key, value*100))
        count+=1

    # print(class_predicted)
    # print(probs)


# # Main Function

# In[10]:

if __name__ == "__main__":
    
    # ----------STEP 1-----------
    # extract_vgg16_features() #Computationally heavy step
    
    # ----------STEP 2-----------
    # model, history = train_top_model()
    
    # ----------STEP 3-----------
    # test_model(model)
    
    # ----------STEP 4-----------
    get_prediction_from_image("../../../dataset/split_data/test/c0/img_42043.jpg")


# In[ ]:




# In[ ]:



