#!/usr/lib/python2.6 python

import sys
from base64 import b64encode as encode
from pickle import dumps
from utilities import MRLogisticRegression, MRData

# Creating model
model = MRLogisticRegression()

# Extracting data
data = MRData(sys.stdin)

# Computing gradient and Hessian
g, H = model.mapper(data.X, data.y)

# Printing mapper results
key = encode(dumps(data.X[0, :]))
value = encode(dumps((g, H)))
print >> sys.stdout, "%s\t%s" % (key, value)
