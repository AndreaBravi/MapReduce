#!/usr/lib/python2.6 python

import numpy as np
from scipy.stats import nanmedian

INF = np.Inf

class MRData(object):
    """

	"""
    def read(self, stream):
        """
        Process the input stream and returns a dataset. It assumes that
        the input data has shape=[n_samples, 1 + n_features], where the first
        column represents the outcome.

        Parameters
        ----------
        stream : stream or file

        Returns
        -------
        X : array, shape=[n_samples, n_features]
            The feature matrix
        y : array, shape=[n_samples, 1]
            The outcome vector
        """
        data = np.genfromtxt(stream, delimiter=',')

        y = np.array(data[:, 0], ndmin=2).T
        X = data[:, 1:]
        return X, y

    def process(self, X, limits=False, normalize=False):
        """
        Process the matrix X

        Values out of the ranges specified in limits are forced to be NaNs, then
        the NaNs are substituted with the median value of the relative feature.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            The feature matrix
        limits : list of tuples or False
                 The list must have len=n_features, each tuple is defined as
                 (minimum acceptable entry, maximal acceptable entry)

        Returns
        -------
        X : array, shape=[n_samples, n_features-1]
            The processed feature matrix
        """
        if limits:
            X = self._nan_out_of_range(X, limits)

        X = self._clean_nans(X)

        if normalize:
            X = self._normalize(X)

        return X

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
    mapper : Computes gradient and Hessian
    reducer : Computes coefficients of the model
	"""
    def mapper(self, X, y, fit_intercept=False):
        """
		Compute gradient and Hessian

		Parameters
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
        if fit_intercept:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        self._check_dimensionality(X, y)
        b = np.zeros([X.shape[1], 1])
        g = self._gradient(X, y, b)
        H = self._hessian(X, y, b)
        return g, H

    def reducer(self, gH):
        """
		Compute coefficients of the model

		Parameters
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
        if y.shape[1] != 1:
            raise MRException('y should be a vector, not a matrix')
        if  X.shape[0] < X.shape[1]:
            raise MRException('More samples are needed to proceed')


class MRException(Exception):
    pass





