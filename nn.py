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
		

 


