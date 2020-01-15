# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np


def svm_naive_loss_function(W, X, y, reg):
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    loss = 0.0
    #############################################################################
    # TODO: Calculez le gradient "dW" et la perte "loss" et stockez le résultat #
    #  dans "dW et dans "loss".                                                 #
    #  Pour cette implementation, vous devez naivement boucler sur chaque pair  #
    #  (X[i],y[i]), déterminer la perte (loss) ainsi que le gradient (voir      #
    #  exemple dans les notes de cours).  La loss ainsi que le gradient doivent #
    #  être par la suite moyennés.  Et, à la fin, n'oubliez pas d'ajouter le    #
    #  terme de régularisation L2 : reg*||w||^2                                 #
    #############################################################################
    nb_class = W.shape[1]
    nb_data = X.shape[0]

    for n in range(nb_data):
      score = np.dot(W.T, X[n,:])
      good_score = score[y[n]]
      for j in range(nb_class):
        if(y[n] == j):
          continue
        bad_score = score[j]   
        margin = bad_score - good_score + 1
        if(margin > 0):
          dW[:,y[n]] -= X[n,:]
          dW[:,j] += X[n,:]
          loss += max(0,margin)
    loss /= nb_data  
    loss += reg*np.linalg.norm(W**2)
    dW /= nb_data

    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW


def svm_vectorized_loss_function(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    nb_class = W.shape[1]
    nb_data = X.shape[0]
    #############################################################################
    # TODO: Implémentez une version vectorisée de la fonction de perte SVM.     #
    # Veuillez mettre le résultat dans la variable "loss".                      #
    # NOTE : Cette fonction ne doit contenir aucune boucle                      #
    #############################################################################
    loss = 0.0
    scores = np.dot(X,W)
    y_scores = np.matrix(scores[np.arange(scores.shape[0]),y]) #this is to get the scores of the good classes.
    marge = np.maximum(0, 1 + scores - y_scores.T)
    marge[np.arange(X.shape[0]),y] = 0
    loss = np.mean(np.sum(marge, axis=1))
    
    #Applying L2 regulation.
    loss += reg*np.linalg.norm(W)**2
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    #############################################################################
    # TODO: Implémentez une version vectorisée du calcul du gradient de la      #
    #  perte SVM.                                                               #
    # Stockez le résultat dans "dW".                                            #
    #                                                                           #
    # Indice: Au lieu de calculer le gradient à partir de zéro, il peut être    #
    # plus facile de réutiliser certaines des valeurs intermédiaires que vous   #
    # avez utilisées pour calculer la perte.                                    #
    #############################################################################

    dW = dW*0

    marge[marge > 0] = 1
    marge[np.arange(X.shape[0]), y] = -1 * np.sum(marge,axis=1).T
    dW = np.dot(X.T, marge)

    dW /= X.shape[0]
    dW += reg * W 
    #############################################################################
    #                            FIN DE VOTRE CODE                              #
    #############################################################################

    return loss, dW
