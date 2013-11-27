#!/usr/lib/python2.6 python

import numpy as np
from scipy.stats import nanmedian

INF = np.Inf

limits = [(0, 1),  # Outcome
          (0, 1),  # Revolving Utilization
          (20, 100),  # Age
          (0, 30),  # Number of times past due 30-59 days
          (0, 1),  # Debt Ratio
          (-1, INF),  # Monthly Income
          (0, 30),  # Number open credit lines
          (0, 10),  # Number of times 90 Days late
          (0, 10),  # Number of real estate loans
          (0, 30),  # Number of times past due 60-89 days
          (0, 50)]  # Number of dependents

class MRData(object):
    def __init__(self, stream):
        data = np.genfromtxt(stream, delimiter=',')
        data = self._nan_out_of_range(data)
        data = self._clean_nans(data)

        self.y = np.array(data[:, 0], ndmin=2).T
        self.X = self._normalize(data[:, 1:])

    def _nan_out_of_range(self, data):
        for col in range(data.shape[1]):
            lmin, lmax = limits[col]
            data[data[:, col] > lmax, col] = np.nan
            data[data[:, col] < lmin, col] = np.nan
        return data

    def _clean_nans(self, data):
        r, c = np.isnan(data).nonzero()

        my = dict()
        for ic in np.unique(c):
            my[ic] = nanmedian(data[:, ic])

        for i in range(len(r)):
            data[r[i], c[i]] = my[c[i]]

        return data

    def _normalize(self, data):
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        return data



class MRLogisticRegression(object):

    def mapper(self, X, y):
        self._check_dimensionality(X, y)
        b = np.zeros([X.shape[1], 1])
        g = self._gradient(X, y, b)
        H = self._hessian(X, y, b)
        return g, H

    def reducer(self, gH):
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








