# using 60, 20, 20 split

import os
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
data_path = './original/'

for i in range(NUM_CLASSES):
    
    curr_dir_path = data_path + 'c' + str(i) + '/'
    
    xtrain = labels = os.listdir(curr_dir_path)
    
    x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)
    
    for x in x_train:
        
        if (not os.path.exists('train/' + 'c' + str(i) + '/')):
            os.makedirs('train/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'train/' + 'c' + str(i) + '/' + x)
        
    for x in x_test:
        
        if (not os.path.exists('test/' + 'c' + str(i) + '/')):
            os.makedirs('test/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'test/' + 'c' + str(i) + '/' + x)
    
    for x in x_val:
        
        if (not os.path.exists('validation/' + 'c' + str(i) + '/')):
            os.makedirs('validation/' + 'c' + str(i) + '/')
            
        os.rename(data_path + 'c' + str(i) + '/' + x, 'validation/' + 'c' + str(i) + '/' + x)