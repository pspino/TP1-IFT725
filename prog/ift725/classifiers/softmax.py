# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def softmax_naive_loss_function(W, X, y, reg):
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
    # TODO: Calculez la perte softmax (entropie croisée) moyenne et son         #
    #  gradient moyen avc des boucles explicites sur chaque paire (X[i], y[i]). #
    #  N'oubliez pas que l'entropie-croisée pour une paire (X[i], y[i]) est     #
    #  -log(SM[y[i]), où SM est le vecteur softmax à 10 classes de X[i]         #
    #  Pour ce qui est du gradient, vous pouvez utiliser l'equation 4.109       #
    #  du livre de Bishop.                                                      #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.                                                         #
    #############################################################################
    wT = W.T
    dat_size = X.shape[0]
    class_size = wT.shape[0]

    for it in range(dat_size):
      real_class = y[it]
      f_t = X[it].dot(wT[real_class])
      e_fj = np.sum(np.exp(wT.dot(X[it])), axis=0)
      loss -= np.log(np.exp(f_t)/e_fj)

      for j in range(class_size):
        dL = np.exp(X[it].dot(wT[j]))
        dL /= e_fj
        if j == real_class:
          dL = dL - 1
        dW.T[j] += X[it] * dL

    loss /= X.shape[0]
    loss += reg * np.linalg.norm(W)**2
    
    dW /= dat_size
    dW += reg * W

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW


def softmax_vectorized_loss_function(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

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
    # TODO: Calculez la perte softmax et son gradient en n'utilisant aucune     #
    #  boucle explicite.                                                        #
    # Stockez la perte dans la variable "loss" et le gradient dans "dW".        #
    # N'oubliez pas la régularisation! Afin d'éviter toute instabilité          #
    # numérique, soustrayez le score maximum de la classe de tous les scores    #
    # d'un échantillon.                                                         #
    #############################################################################
    #Loss
    scores_prob = np.exp(X.dot(W)) / np.sum(np.exp(X.dot(W)), axis=1, keepdims=True)
    good_score = scores_prob[range(X.shape[0]),y]
    loss = np.sum(-np.log(good_score))
    loss /= X.shape[0]
    loss += reg * np.linalg.norm(W)**2

    #Gradient
    dW = np.exp(X.dot(W)) / np.sum(np.exp(X.dot(W)), axis=1, keepdims=True)
    dW[range(X.shape[0]),y] -= 1
    dW /= X.shape[0]
    dW = X.T.dot(dW)  
    dW += reg * W

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW
