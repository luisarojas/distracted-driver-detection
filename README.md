### Dependencies

`Python 3.6.1`
`Tensorflow 1.3.0`
`Keras 2.1.2`
`matplotlib 2.0.2`
`numpy 1.12.1`

### Run the Code

* **Simple CNN in Keras**

	Directory Path: `/src/keras/base`

	* **Train the model**: `python train.py`
	
	* **Test the model:**: `python test.py`
	
	* **Predict from an image**: `predict.py [-h] [--image IMAGE] [--hide_img]`, where the `image` flag is to be followed by the path to an image *(optional)*; if none is provided, the default will be used. The `hide_flag`  is to avoid the display of the image on termination *(optional)*.

* **CNN with VGG16 Transfered Learning**

	Directory Path: `/src/keras/vgg`

	* **Train the model**: `python train_top.py`
	
	* **Test the model:**: `python test.py [-h] [--acc] [--cm] [--roc]`, where at least one must be used. `acc` will output the total accuracy of the model, `cm` will draw a confusion matrix, and `roc` will plot the receiver operating characteristic for this model.
	
	* **Predict from an image**: `predict.py [-h] [--image IMAGE] [--hide_img]`, where the `image` flag is to be followed by the path to an image *(optional)*; if none is provided, the default will be used. The `hide_img` flag is to avoid the display of the image on termination *(optional)*.

# Detecting Distracted Drivers

## Objective

Predict the likelihood of what the driver is doing in each of the pictures in the dataset.

## Dataset

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

The dataset consists on a set of images, each taken in a car where the driver is doing some action (e.g. texting, talking on the phone, doing their makeup). These are some examples:

<img src="./readme_res/1.jpg" width=200> <img src="./readme_res/2.jpg" width=200> <img src="./readme_res/3.jpg" width=200>

The images are labeled following a set of 10 categories:

|Class|Description|
|-----|-----------|
| `c0` | Safe driving. |
| `c1` | Texting (right hand). |
| `c2` | Talking on the phone (right hand). |
| `c3` | Texting (left hand). |
| `c4` | Talking on the phone (left hand). |
| `c5` | Operating the radio. |
| `c6` | Drinking. |
| `c7` | Reaching behind. |
| `c8` | Hair and makeup. |
| `c9` | Talking to passenger(s). |