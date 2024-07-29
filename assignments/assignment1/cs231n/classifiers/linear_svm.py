from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    C = W.shape[1]
    D = W.shape[0]
    N = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    beta = np.zeros((N, C))

    # compute the loss and the gradient
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(C):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if (margin > 0):
                beta[i][j] =1.0
                loss += margin
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    third = 0.0
    for d in range(D):
        for c in range(C):
            first = 0.0
            for n in range(N):
                beta_dummy = beta[n,c] 
                for j in range(C):
                    beta[n,c] -= beta[n,j]
                first = beta_dummy * X[n,d]
            dW[d,c] = first/N + 2*reg*W[d,c]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    C = W.shape[1]
    N = X.shape[0]
    loss = 0.0 
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    beta = np.zeros((N, C))
    margin = np.zeros((N, C))
    one_hot_encoded_y = np.eye(C)[y]   # one_hot_encoded_y data shape: 500 * 10
    first_term = np.ones((N,C)) - one_hot_encoded_y
    third_term = X @ W
    fourth_term = third_term[np.arange(N), y]
    delta_ones = np.zeros((N, C))
    margin = third_term - fourth_term[:, np.newaxis] + delta_ones
    beta = (margin > 0).astype(int)

    # compute the loss and the gradient

    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # Add regularization to the loss.
    loss += np.sum(first_term * beta * margin)/N + reg*np.sum(W**2)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    first = X.T @ beta  # first shape: (D, C)
    print("first shape: ", first.shape)
    sum_beta = np.sum(beta, axis=1) # shape: (N, 1)
    print("sum_beta shape: ", sum_beta.shape)
    second = np.sum(sum_beta[:, np.newaxis] * X, axis = 0) # second shape: (D,)
    print("second shape: ", second.shape)
    second_extended = first - second[:, np.newaxis]
    print("second extended shape: ", second_extended.shape)
    third = 2*reg*W # third shape: (D, C)
    print("third shape: ", third.shape)
    dW = (first - second[:, np.newaxis])/N + third

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
