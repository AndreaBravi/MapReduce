#!/usr/lib/python2.6 python

import numpy as np
from scipy.stats import nanmedian

INF = np.Inf

# Pre-specified limits of the variables 
limits = [(0, 1),  # Outcome
          (0, 1),  # Revolving utilization
          (20, 100),  # Age
          (0, 30),  # Number of times 30-59 days past due 
          (0, 1),  # Debt ratio
          (-1, INF),  # Monthly income
          (0, 30),  # Number open credit lines
          (0, 10),  # Number of times 90 days or more past due 
          (0, 10),  # Number of real estate loans
          (0, 30),  # Number of times 60-89 days past due 
          (0, 50)]  # Number of dependents

class MRData(object):
	"""
	Process input stream and returns a clean dataset.		   
	"""
    def __init__(self, stream):
	    """
        Parameters
        ----------
        self : stream or file
            Input data        
        """
        data = np.genfromtxt(stream, delimiter=',')
        data = self._nan_out_of_range(data)
        data = self._clean_nans(data)

        self.y = np.array(data[:, 0], ndmin=2).T
        self.X = self._normalize(data[:, 1:])

    def _nan_out_of_range(self, data):
		"""
		Place a NaNs in those elements of the data matrix that are
		outside of the values specified in limits
		
        Parameters
        ----------
        data : array, shape=[n_samples, n_features]
            Data array
        """
        for col in range(data.shape[1]):
            lmin, lmax = limits[col]
            data[data[:, col] > lmax, col] = np.nan
            data[data[:, col] < lmin, col] = np.nan
        return data

    def _clean_nans(self, data):
		"""
		Substitute NaNs with the median value of the related features
		
        Parameters
        ----------
        data : array, shape=[n_samples, n_features]
            Data array
        """
        r, c = np.isnan(data).nonzero()

        my = dict()
        for ic in np.unique(c):
            my[ic] = nanmedian(data[:, ic])

        for i in range(len(r)):
            data[r[i], c[i]] = my[c[i]]

        return data

    def _normalize(self, data):
		"""
		Subtract the mean from data, and divide by the standard
		deviation, along the direction of the features.
		
        Parameters
        ----------
        data : array, shape=[n_samples, n_features]
            Data array
        """
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        return data



class MRLogisticRegression(object):
	"""
	Logistic regression for MapReduce
	
	Methods
    ----------
    k : int, default=2
        Number of features to select (selected_features)
    mask : list, len=selected_features
           Integer list of the features ordered by maximal relevance and
           minimal redundancy
    score : array, shape=[selected_features]
            mRMR score associated to each entry in 
	"""
    def mapper(self, X, y):
		"""
		Compute gradient and Hessian
		
		Attributes
		----------
		X : array, shape=[n_samples, n_features]
			Feature matrix
		y : array, shape=[n_samples, 1]
			Outcome vector
			
		Returns
		-------
		g : array, shape=[n_features, 1]
			Gradient vector
		H : array, shape=[n_features, n_features]
			Hessian matrix
		"""
        self._check_dimensionality(X, y)
        b = np.zeros([X.shape[1], 1])
        g = self._gradient(X, y, b)
        H = self._hessian(X, y, b)
        return g, H

    def reducer(self, gH):
		"""
		Compute coefficients of the logistic regression
		
		Attributes
		----------
		gH : list of tuples, len(gH)=Number of map jobs
			Each element of the list contains a tuple of the type (g, H)
			where g is the gradient, H is the Hessian matrix		
			
		Returns
		-------
		b : array, shape=[n_features, 1]
			Coefficients of the logistic regression		
		"""
        b = np.zeros([gH[0][0].shape[0], 1])
        for g, H in gH:
            H_1 = np.linalg.inv(H)
            b -= np.dot(H_1, g)
        return b

    def _gradient(self, X, y, b):
        p = self._sigmoid(X, b)
        return np.dot(X.T, y - p)

    def _hessian(self, X, y, b):
        p = self._sigmoid(X, b)
        W = np.identity(X.shape[0]) * p
        return -np.dot(np.dot(X.T, W), X)

    def _sigmoid(self, x, b):
        return 1 / (1 + np.exp(-np.dot(x, b)))

    def _check_dimensionality(self, X, y):
        if y.shape[1] != 1 or X.shape[0] < X.shape[1]:
            return -1








