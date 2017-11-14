# MCSC6230G Advanced Topics in High-Performance Computing - Project

## Details

### Deliverables

**Description**: Focus on how Machine Learning is applied to a problem, rather than the actual implementation. Work in pairs is allowed.

**Proposal**: Due Oct. 31st, 2017.
**Finished Project Presentation**: November 29th, 2017
**Finished Project Report**: December 8th, 2017

### Objective

Predict the likelihood of what the driver is doing in each of the pictures in the   dataset.

### Method

We will split the data, where 70% will correspond to the training set, and the remaining 30% will correspond to the test set. After, the ratio can be adjusted as we see fit, depending on the network’s behaviour.

First, we will implement a simple Feedforward Neural Network with a single hidden layer. In order to minimize the change of overfitting, we will apply regularization to find the optimal 2 value.

Once this value has been found, we will increase the number of hidden layers. After, the number of hidden layers will be increased depending on it’s performance. It’s important to note that this will involve an unknown upper bound, since we are limited by computational power.

Finally, we will implement a Convolutional Neural Network and observe how it performs against the aforementioned methods. The differences of the methods applied will be illustrated through several graphs in terms of efficiency and accuracy. Our hypothesis is that this method will outperform the rest and will be accurate enough to qualify for the competition’s leaderboards.

We will accomplish the above using Python as our programming language and either Tensorflow or Keras as our framework, or using the Octave programming language.

## Resources

### Data Sets

| Description | Link |
| ----------- | ---- |
| Kaggle: "State Farm Distracted Driver Detection" | https://www.kaggle.com/c/state-farm-distracted-driver-detection/data |

### Machine Learning

| Tool | Link |
| ---- | ---- |
| TensorFlow | https://www.tensorflow.org |
| Keras | https://keras.io |
| Octave | https://www.gnu.org/software/octave |

