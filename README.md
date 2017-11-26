# MCSC6230G Advanced Topics in High-Performance Computing - Project

## Details

### Deliverables

**Description**: Focus on how Machine Learning is applied to a problem, rather than the actual implementation. Work in pairs is allowed.

<strike>**Proposal**: Due Oct. 31st, 2017.</strike>

**Project Presentation**: November 29th, 2017
> 10 minutes (7 minute presentation + demo, 3 min QA and discussion)

**Project Report**: December 8th, 2017

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

### Resources

* **Generic**

	* On Neural Networks:

		https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

	* On CNN Architectures:
	
		https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
		https://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/
		https://wiki.tum.de/display/lfdv/Convolutional+Neural+Network+Architectures#ConvolutionalNeuralNetworkArchitectures-AlexNet
	
	* On CNN Implementations
		
		http://cs231n.github.io/
		https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
		https://benanne.github.io/2015/03/17/plankton.html

* **Papers**

	https://pdfs.semanticscholar.org/cb49/ac9618bb2f8271409f91d53254a095d843d5.pdf

* **Other**

	https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/22631