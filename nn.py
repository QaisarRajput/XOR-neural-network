# The program creates an neural network that simulates the exclusive OR function with two inputs and one output.

import numpy as np # cuz swag

'''
We define a sigmoid function. This function runs in every neuron of the network
when data hits it. Sigmoid functions are widely used when building a neural
network because they introduce non-linearity in the model. Probabilities out of
numbers can thus be generated easily.

A sigmoid function is in the form 1 / ( 1 + e ^ (-x) ), where x is the input.

'''
def nonlin(x, derivative = False):
	if (derivative == true):
		return ( x * ( x -1 ))
	
	return (1 / (1 + np.exp(-x)))
		
# Initialize the input data set as a matrix
# Each row is a different training example
# Each column is a different neuron
# The third column represents the bias term and is not part
# of the input
X = np.array([	[0, 0, 1],
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 1]
		])

# Intialize the output data set as a matrix
y = np.array ([	[0],
		[1],
		[1],
		[0]
		])

# To generate random numbers, we need to seed them
# to make them deterministic. ie return the same set of random numbers
# each time the program runs.
# This is useful for debugging.
np.random.seed(1)

# Create synapse matrices
# Synapses are the connections between each neuron in one layer to 
# every neuron in the next layer.
'''
The weights are intialized to random values.
Since this example has 3 layers: Input, Hidden and Output. It would require 2 synapses 

syn0 are the weights between the input layer and the hidden layer.  
It is a 3x4 matrix because there are two input weights plus a bias term (=3) and 
four nodes in the hidden layer (=4). 

syn1 are the weights between the hidden layer and the output layer. 
It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. 

Note that there is no bias term feeding the output layer in this example. 

The weights are initially generated randomly because optimization tends 
not to work well when all the weights start at the same value. 

'''
syn0 = 2 * np.random.random((3, 4)) - 1 # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2 * np.random.random((4, 1)) - 1 # 4x1 matrix of weights (4 nodes x 1 output)
 


 


