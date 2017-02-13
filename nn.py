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
 
# Training step.
# We use a for-loop to iterate through the training code to optimise the network for the given data set
for j in xrange(60000):
	# Calculates forward through the network
	'''
	First, we create the first layer. ie Input Layer
	The prediction step involves performing matrix multiplication
	between each layer and its synapse.
	Next, we run the sigmoid function on all the values in the matrix
	to create the next layer. The next layer contains a prediction of 
	the output data.
	Repeat the same process to get the next layer, which is a more refined prediction.

	'''
	l0 = X
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))
	
	# We compare the predictions of the output value in layer 2 to 
	# the expected output data using subtraction to get the error rate.
	l2_error = y - l2

        # It is also helpful to print out the average error rate at a set interval
        # to make sure it goes down every time
        if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
                print "Error: " + str(np.mean(np.abs(l2_error)))	
	
	# Back propagation of errors using the chain rule.
	'''
	Multiply the error rate by the result of our sigmoid function.
	The function is used to get the derivative of our output predictions from 
	layer two.
	We store this result in delta which is used to minimize error rate of the prediction
	when we update our synapses every iteration.
	Next, we want to see how much layer one contributed to the error in layer two.
	This is acheived using backpropagation. 
	'''

