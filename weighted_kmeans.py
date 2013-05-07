import warnings

from numpy.random import randint
from numpy import shape, zeros, sqrt, argmin, minimum, array,\
    newaxis, arange, compress, equal, common_type, take,\
    std, mean, average
import numpy as np
from scipy.cluster import vq

def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, weights=None):
    """
    Performs k-means on a set of observation vectors forming k clusters.

    The k-means algorithm adjusts the centroids until sufficient
    progress cannot be made, i.e. the change in distortion since
    the last iteration is less than some threshold. This yields
    a code book mapping centroids to codes and vice versa.

    Distortion is defined as the sum of the squared differences
    between the observations and the corresponding centroid.

    Parameters
    ----------
    obs : ndarray
       Each row of the M by N array is an observation vector. The
       columns are the features seen during each observation.
       The features must be whitened first with the `whiten` function.

    k_or_guess : int or ndarray
       The number of centroids to generate. A code is assigned to
       each centroid, which is also the row index of the centroid
       in the code_book matrix generated.

       The initial k centroids are chosen by randomly selecting
       observations from the observation matrix. Alternatively,
       passing a k by N array specifies the initial k centroids.

    iter : int, optional
       The number of times to run k-means, returning the codebook
       with the lowest distortion. This argument is ignored if
       initial centroids are specified with an array for the
       ``k_or_guess`` parameter. This parameter does not represent the
       number of iterations of the k-means algorithm.

    thresh : float, optional
       Terminates the k-means algorithm if the change in
       distortion since the last k-means iteration is less than
       or equal to thresh.

    Returns
    -------
    codebook : ndarray
       A k by N array of k centroids. The i'th centroid
       codebook[i] is represented with the code i. The centroids
       and codes generated represent the lowest distortion seen,
       not necessarily the globally minimal distortion.

    distortion : float
       The distortion between the observations passed and the
       centroids generated.

    See Also
    --------
    kmeans2 : a different implementation of k-means clustering
       with more methods for generating initial centroids but without
       using a distortion change threshold as a stopping criterion.

    whiten : must be called prior to passing an observation matrix
       to kmeans.

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.cluster.vq import vq, kmeans, whiten
    >>> features  = array([[ 1.9,2.3],
    ...                    [ 1.5,2.5],
    ...                    [ 0.8,0.6],
    ...                    [ 0.4,1.8],
    ...                    [ 0.1,0.1],
    ...                    [ 0.2,1.8],
    ...                    [ 2.0,0.5],
    ...                    [ 0.3,1.5],
    ...                    [ 1.0,1.0]])
    >>> whitened = whiten(features)
    >>> book = array((whitened[0],whitened[2]))
    >>> kmeans(whitened,book)
    (array([[ 2.3110306 ,  2.86287398],
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)

    >>> from numpy import random
    >>> random.seed((1000,2000))
    >>> codes = 3
    >>> kmeans(whitened,codes)
    (array([[ 2.3110306 ,  2.86287398],
           [ 1.32544402,  0.65607529],
           [ 0.40782893,  2.02786907]]), 0.5196582527686241)

    """
    if int(iter) < 1:
        raise ValueError('iter must be at least 1.')
    if type(k_or_guess) == type(array([])):
        guess = k_or_guess
        if guess.size < 1:
            raise ValueError("Asked for 0 cluster ? initial book was %s" %\
                             guess)
        result = _kmeans(obs, guess, thresh = thresh, weights=weights)
    else:
        #initialize best distance value to a large value
        best_dist = np.inf
        No = obs.shape[0]
        k = k_or_guess
        if k < 1:
            raise ValueError("Asked for 0 cluster ? ")
        for i in range(iter):
            #the intial code book is randomly selected from observations
            guess = take(obs, randint(0, No, k), 0)
            book, dist = _kmeans(obs, guess, thresh = thresh, weights=weights)
            if dist < best_dist:
                best_book = book
                best_dist = dist
        result = best_book, best_dist
    return result

def _kmeans(obs, guess, thresh=1e-5, weights=None):
    """ "raw" version of k-means.

    Returns
    -------
    code_book :
        the lowest distortion codebook found.
    avg_dist :
        the average distance a observation is from a code in the book.
        Lower means the code_book matches the data better.

    See Also
    --------
    kmeans : wrapper around k-means

    XXX should have an axis variable here.

    Examples
    --------
    Note: not whitened in this example.

    >>> from numpy import array
    >>> from scipy.cluster.vq import _kmeans
    >>> features  = array([[ 1.9,2.3],
    ...                    [ 1.5,2.5],
    ...                    [ 0.8,0.6],
    ...                    [ 0.4,1.8],
    ...                    [ 1.0,1.0]])
    >>> book = array((features[0],features[2]))
    >>> _kmeans(features,book)
    (array([[ 1.7       ,  2.4       ],
           [ 0.73333333,  1.13333333]]), 0.40563916697728591)

    """

    code_book = array(guess, copy = True)
    avg_dist = []
    diff = thresh+1.
    while diff > thresh:
        nc = code_book.shape[0]
        #compute membership and distances between obs and code_book
        obs_code, distort = vq.vq(obs, code_book)
        avg_dist.append(mean(distort, axis=-1))
        #recalc code_book as centroids of associated obs
        if(diff > thresh):
            has_members = []
            for i in arange(nc):
                cell_members = compress(equal(obs_code, i), obs, 0)
                if cell_members.shape[0] > 0:
                    if not weights:
                        code_book[i] = mean(cell_members, 0)
                    else:
                        code_book[i] = average(cell_members, 0, weights)
                    has_members.append(i)
                #remove code_books that didn't have any members
            code_book = take(code_book, has_members, 0)
        if len(avg_dist) > 1:
            diff = avg_dist[-2] - avg_dist[-1]
        #print avg_dist
    return code_book, avg_dist[-1]