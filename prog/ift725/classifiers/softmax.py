# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson
# et autres. Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

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

    nb_data, nb_class = X.shape[0], W.shape[1]

    for n in range(nb_data):
        f = np.dot(X[n, :], W)
        scores = np.exp(f - np.max(f))  # subtract maximum class score for numerical stability
        for k in range(nb_class):
            p = scores[k] / np.sum(scores)
            dscore = p
            # decrease loss on correct class
            if k == y[n]:
                dscore -= 1
                loss -= np.log(p)
            dW[:, k] += X[n, :] * dscore

    loss /= nb_data
    loss += reg * np.linalg.norm(W ** 2)
    dW /= nb_data

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

    nb_data, nb_class = X.shape[0], W.shape[1]

    f = np.dot(X, W)
    scores = np.exp(f - np.max(f))  # subtract maximum class score for numerical stability
    prob = scores / np.sum(scores, axis=1, keepdims=True)
    good_scores = -np.log(prob[np.arange(X.shape[0]), y])

    loss = np.sum(good_scores)
    dscore = prob
    dscore[np.arange(X.shape[0]), y] -= 1

    dW = np.dot(X.T, dscore)

    loss /= nb_data
    loss += reg * np.linalg.norm(W ** 2)
    dW /= nb_data

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW