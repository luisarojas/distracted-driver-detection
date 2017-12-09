
# ---------- HANDLE COMMAND LINE ARGUMENTS ----------

import argparse
import sys

parser = argparse.ArgumentParser(description="Predict the class of a given driver image. You must select at least one metric to display: [acc, cm, roc]")
parser.add_argument("--acc", action="store_true", help="will calculate loss and accuracy")
parser.add_argument("--cm", action="store_true", help="will plot confusion matrix")
parser.add_argument("--roc", action="store_true", help="will plot roc curve")

# exit if no arguments are passed
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
    
# otherwise, continue
args = parser.parse_args()

from keras.preprocessing.image import ImageDataGenerator
from helper import target_size, batch_size, num_classes, create_top_model, class_labels, class_labels_encoded
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

# ---------- FUNCTION DEFINITIONS ----------

def plot_confusion_matrix(y_true, y_pred):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    title = 'Confusion matrix'
    
    # normalize matrix
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels_encoded))
    plt.xticks(tick_marks, class_labels_encoded, rotation=0)
    plt.yticks(tick_marks, class_labels_encoded)

    threshold = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > threshold else 'black')

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc(y_true, y_pred):
    
    # calculate roc and auc
    false_pos_rate = dict()
    true_pos_rate = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        false_pos_rate[i], true_pos_rate[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        roc_auc[i] = auc(false_pos_rate[i], true_pos_rate[i])
        
    # plot all
    
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(false_pos_rate[i], true_pos_rate[i], lw=2, c=color,
                    label='c{0} (auc = {1:0.2f})'.format(i, roc_auc[i]))

    # plot random guess roc
    plt.plot([0, 1], [0, 1], 'k--',color='salmon', lw=2, label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    
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
test_labels_onehot = to_categorical(test_labels, num_classes=num_classes) # class number in onehot

# ---------- TEST MODEL ----------

model = create_top_model("softmax", test_data.shape[1:])
model.load_weights("res/_top_model_weights.h5")  

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

predicted = model.predict_classes(test_data)

# ---------- DISPLAY INFORMATION DEPENDING ON COMMAND LINE FLAGS ----------

if args.acc:
    loss, acc = model.evaluate(test_data, test_labels_onehot, batch_size=batch_size, verbose=1)
    print("loss: ", loss)
    print("accuracy: {:8f}%".format(acc*100))

if args.cm:
    plot_confusion_matrix(test_labels, predicted)
    
if args.roc:
    predicted_onehot = to_categorical(predicted, num_classes=num_classes)
    plot_roc(test_labels_onehot, predicted_onehot)