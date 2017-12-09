from keras.preprocessing.image import ImageDataGenerator
from helper import target_size, batch_size, num_classes, create_top_model, class_labels, class_labels_encoded
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def get_confusion_matrix(y_true, y_pred, class_names):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    title = 'Confusion matrix'
    
    # normalize matrix
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    threshold = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > threshold else 'black')

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# ---------- GET TESTING DATA ----------

# create datagen and train generator to load the data from directory
datagen = ImageDataGenerator(rescale=1.0/255.0) 
test_generator = datagen.flow_from_directory(
                        '../../../dataset/split_data/test/', 
                        target_size=target_size,
                        batch_size=batch_size,
                        class_mode='categorical', # specify categorical
                        shuffle=False) # data is ordered

# load vgg features
test_data = np.load('res/vgg_test_features.npy')

test_labels = test_generator.classes # actual class number
# test_labels_onehot = to_categorical(test_labels, num_classes=num_classes) # class number in onehot

# ---------- TEST MODEL ----------

model = create_top_model("softmax", test_data.shape[1:])
model.load_weights("res/_top_model_weights.h5")  

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# loss, acc = model.evaluate(test_data, test_labels_onehot, batch_size=batch_size, verbose=1)

# print()
# print("loss: ", loss)
# print("accuracy: {:8f}%".format(acc*100))

# predicted = model.predict(test_data)
predicted = model.predict_classes(test_data)

# convert to array like [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# predicted_onehot = np.zeros_like(predicted)
# predicted_onehot[np.arange(len(predicted)), predicted.argmax(1)] = 1

get_confusion_matrix(test_labels, predicted, class_labels_encoded)