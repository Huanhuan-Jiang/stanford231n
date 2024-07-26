from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=X.shape[0]
    D=X.shape[1]
    C=W.shape[1]
    f = np.zeros((C, N))    #f[c][n] is the score for class c for the n-th sample
    yy = np.zeros((C,N))    #yy_=[c][n] is the probability for class c for the n-th sample
    sum_yy = np.zeros(N)    #sum[n] is the sum of all y[c][n] over all classes
    R = 0.0

    for n in range(N):
       for c in range(C):
          for d in range(D):
             f[c][n] += W[d][c]*X[n][d]
    
    for n in range(N):
       for c in range(C):
          sum_yy[n] += np.exp(f[c][n])
       for c in range(C):
          yy[c][n] = np.exp(f[c][n])/sum_yy[n]

    for d in range(D):
       for c in range(C):
          R += W[d][c]**2
    R *= reg
    
    for n in range(N):
       k = y[n]
       loss += -np.log(yy[k][n])   # the loss of the n-th sample
    loss /= N
    loss += R
        


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
