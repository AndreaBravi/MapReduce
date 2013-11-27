#!/usr/lib/python2.6 python

import sys
from base64 import b64encode as encode
from pickle import dumps
from utilities import MRLogisticRegression, MRData

# Pre-specified limits of the variables
limits = [(0, 1),  # Outcome
          (0, 1),  # Revolving utilization
          (20, 100),  # Age
          (0, 30),  # Number of times 30-59 days past due
          (0, 1),  # Debt ratio
          (0, INF),  # Monthly income
          (0, 30),  # Number open credit lines
          (0, 10),  # Number of times 90 days or more past due
          (0, 10),  # Number of real estate loans
          (0, 30),  # Number of times 60-89 days past due
          (0, 50)]  # Number of dependents

# Creating model
model = MRLogisticRegression()

# Extracting data
data = MRData()
X, y = data.read(sys.stdin)
X = data.process(data, limits=limits, normalize=True)

# Computing gradient and Hessian
g, H = model.mapper(X, y)

# Printing mapper results
key = encode(dumps(X[0, :]))
value = encode(dumps((g, H)))
print >> sys.stdout, "%s\t%s" % (key, value)
