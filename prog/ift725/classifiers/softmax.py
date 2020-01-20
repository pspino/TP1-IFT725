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
    loss = loss*0
    dW = dW*0
    
    N, C = X.shape[0], W.shape[1]
    dWT = np.transpose(dW)
    for i in range(N):
        f_i = np.dot(X[i], W)
        f_i -= np.max(f_i)

        t_i = y[i]
        f_t = f_i[t_i]
        p_f_i = np.sum(np.exp(f_i))
        S_i = np.exp(f_t) / p_f_i

        loss += -np.log(S_i)

        for j in range(C):
            if j == t_i:
                dWT[t_i] += (np.exp(f_i[j]) / p_f_i - 1) * X[i]
            else:
                dWT[j] += (np.exp(f_i[j]) / p_f_i) * X[i]
    
    # normalization
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N
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
    loss = loss * 0
    dW = dW * 0

    N = X.shape[0]
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    row_sum = exp_scores.sum(axis=1)
    row_sum = row_sum.reshape((N, 1))

    norm_exp_scores = exp_scores / row_sum
    row_index = np.arange(N)
    loss = norm_exp_scores[row_index, y].sum()
    norm_exp_scores[row_index, y] -= 1

    dW = np.dot(np.transpose(X), norm_exp_scores)

    loss /= N
    loss += reg * np.sum(W * W)
    dW /= N
    dW += reg * W

    #############################################################################
    #                         FIN DE VOTRE CODE                                 #
    #############################################################################

    return loss, dW
