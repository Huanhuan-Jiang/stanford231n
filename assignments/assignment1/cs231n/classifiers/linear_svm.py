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
    for n in range(N):
        scores = X[n].dot(W)
        correct_class_score = scores[y[n]]
        for c in range(C):
            if c == y[n]:
                continue
            margin = scores[c] - correct_class_score + 1  # note delta = 1
            if (margin > 0):
                #beta[n][c] =1.0
                loss += margin
                dW[:, c] += X[n]
                dW[:, y[n]] -= X[n]
    
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
    dW /= N 
    dW += 2*reg*W

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
    delta = 1.0
    beta = np.zeros((N, C))
    margins = np.zeros((N, C))

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    scores_true = scores[np.arange(N), y]
    margins = np.maximum(0, scores - scores_true[:, np.newaxis] + delta)
    margins[np.arange(N),y] = 0.0

    loss = np.sum(margins)/N + reg*np.sum(W**2)
    
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
    beta = (margins > 0).astype(float)
    beta[np.arange(N), y] -= np.sum(beta, axis=1) 
    dW = X.T @ beta/N + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
