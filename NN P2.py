"""
    This is the second part of the neural network code
    3 Layer Neural Network - MORE COMPLICATED
    Neither column has any correlation to the output, 50% chance of getting a 0 or a 1
    Pattern is that if either column (but not both) is a one the output is a 1
    Non-linear as it is a one-to-one relationship between a combination of inputs.

    X :     Input Dataset matrix where each row is a training example
    y :     Output Dataset matrix where each row is a training example
    I0 :    First layer of Network, specified by input data
    I1 :    Second layer of Network, otherwise known as the hidden layer
    I2 :    Final layer of Network, which is our hypothesis, and should approximate the correct answer as we train
    syn0 :  First layer of weights, Synapse 0, connecting I0 to I1
    syn1 :  Second layer of weights, Synapse 1, connecting I1 to I2
    I2_error :  Amount that the neural network 'missed'
    I2_delta :  This is the error of the network scaled by the confidence. Almost identical to the error except that
                very confident errors are muted
    I1_error :  Weighting I2_delta by the weights in syn1, we can calculate the error in the I1
    I1_delta :  This is the I1 error of the network scaled by the confidence. Almost identical to the I1 error except
                that very confident errors are muted
"""

import numpy as np  # imports linear algebra library


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j % 10000) == 0:
        print("Error:", str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
        # uses the 'confidence weighted error' from I2 to establish an error for I1. Ir do this it sends the error
        # across the weights from I2 to I1. This gives what is called a 'contribution weighted error' because we learn
        # how much each node value in I1 contributed to the error in I2. This is called BACKPROPAGATING.
        # then update syn0 using the same steps we did in the 2 layer implementation
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print(l1)
print(l2)
