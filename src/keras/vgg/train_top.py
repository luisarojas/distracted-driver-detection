from helper import create_top_model, num_classes, class_labels, target_size, batch_size

# global variables
epochs = 5

class_labels_onehot = to_categorical(class_labels, num_classes)
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

if debug:
    print('>>> train_generator.classes' + str(train_generator.classes))

# load vgg features
train_data = np.load('vgg_train_features.npy')

# ---------- LOAD VALIDATION DATA ----------
# create datagen and train generator to load the data from directory

val_generator = datagen.flow_from_directory(
                            '../../../dataset/split_data/validation/',
                            target_size=target_size,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=False) # data is ordered
                            
num_val_samples = len(val_generator.filenames)

if debug:
    print('>>> val_generator.classes' + str(val_generator.classes))

# load vgg features
val_data = np.load('vgg_val_features.npy')

# ---------- CREATE AND TRAIN MODEL ----------

if debug:
    print('>>> tran_data.shape (check [1:]) : ' + str(train_data.shape))

# create the top model to be trained
model = create_top_model("softmax", train_data.shape[1:])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# only save the best weights. if the accuracy doesnt improve in 2 epochs, stop.
checkpoint_callback = ModelCheckpoint(
                        "top_model_weights.h5", # store weights with this file name
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
            # train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            # validation_data=(val_data, val_labels)
            callbacks=callbacks_list)

