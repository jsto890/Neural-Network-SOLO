"""
    This is the first part of the neural network code
    2 Layer Neural Network

    X :     Input Dataset matrix where each row is a training example
    y :     Output Dataset matrix where each row is a training example
    I0 :    First layer of Network, specified by input data
    I1 :    Second layer of Network, otherwise known as the hidden layer
    syn0 :  First layer of weights, Synapse 0, connecting I0 to I1
    x.dot(y) :  If vectors this is dot product,
                If matrix this is matrix-matrix multiplication,
                If only one is a matrix then it is vector-matrix multiplication
"""

import numpy as np  # imports linear algebra library


# sigmoid function
def nonlin(x, deriv=False):  # def nonlin(x, deriv=False): 'nonlinearity', maps a function called a sigmoid, maps any
    # value to a value between 0 and 1. Use it to convert numbers to probability
    if deriv:  # if deriv: Can generate derivative of a sigmoid (when deriv=True). A sigmoid functions output can be
        # used to create its derivative. if the output is 'x' the derivative is 'x'*(1 - 'x')
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
    # initialises our input dataset as a numpy matrix. Each row is a single 'training example'. Each column
    # corresponds to one of our input nodes and each row is a training example.
    # So, we have 3 input nodes to the network and 4 training examples
X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

# output dataset
    # Initialises our output dataset, generated the data set horizontally. '.T' is to transpose the matrix, after
    # the transpose it has 4 rows with one column. We use the inputs from the input and use the new transposed
    # matrix columns as the output node. So, 3 inputs and 1 output
y = np.array([[0, 1, 1, 0]]).T

# seed random numbers to make calculation
    # deterministic (just a good practice)
    # good idea to seed your random numbers, will be randomly distributed but will be randomly distributed in the
    # exact same way each time you train, makes it easier to see how changes affect the network
np.random.seed(1)

# initialize weights randomly with mean 0
    # weight matrix for neural network, syn0 to imply synapse zero. since we only have two layers (input and output)
    # we only need one matrix of weights to connect them. Dimension is (3,1) because we have 3 inputs and one output.
    # I0 is of size 3 and I1 is of size 1, thus we want to connect every node in I0 to every node in I1.
    # Best practice to have a mean weight of zero in weight initialisation
syn0 = 2 * np.random.random((3, 1)) - 1

#  Actual network training code begins, loop iterated multiple times over the training code to optimise our
#  network to the dataset
for iter in range(10000):

    # forward propagation
    l0 = X  # Since our first layer I0 is just the data we describe it as such, X contains 4 training examples (rows)
            # We process them all at the same time (called 'full batch' training)
            # Thus we have 4 different I0 rows, but is thought of as only one training example (row)
    l1 = nonlin(np.dot(l0, syn0))   # Prediction step, first we let the network 'try' to predict the output given the
                                    # input, See how it performs so that we can adjust it to do a bit better
                                    # next iteration. Line has 2 steps, first multiplies I0 by syn0 and the second
                                    # passes the output through the sigmoid function. Matrix ends up as a (4x1).
                                    # End matrix (a x b) is number of rows of first matrix (a) and columns in second (b)
                                    # Since we loaded 4 training examples we end up with 4 guesses for a correct answer

    # how much did we miss?
        # subtract true answer (y) from the guess (l1), output is just a vector of positive and negative numbers
        # reflecting how much the network missed
    l1_error = y - l1

    # multiply how much we missed by the slope of the sigmoid at the values in l1
        # nonlin(l1, True) : code generated slopes for certain points on the function line.
            # Very high and Very Low have shallow slopes while x = 0 (middle) has the steepest slope
        # l1_delta = l1_error * nonlin(l1, True) : 'error (4 x 1)' times 'slopes (4 x 1)', we reduce the error of the
            # high confidence predictions. points with lower slope have more confidence in their accuracy.
            # updates the central points more heavily
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    # computes the weight updates for each weight for each training example, sums them and updates all the weight
    syn0 += np.dot(l0.T, l1_delta)

# Output After Training:
print("Output After Training:")
print(l1)


