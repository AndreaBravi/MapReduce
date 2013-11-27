#!/usr/lib/python2.6 python

import sys
from utilities import MRLogisticRegression
from base64 import b64decode as decode
from pickle import loads

# Creating model
model = MRLogisticRegression()

# Aggregating all the gradients and Hessians
g_H=[]
for line in sys.stdin:
    line = line.strip()
    key, value = line.split("\t", 1)
    g_H.append(loads(decode(value)))
	
# Computing model coefficients
beta = model.reducer(g_H)

# Printing reducer results
for i in range(len(beta)):
    key = i
    value = beta[i]
    print >> sys.stdout, "%s\t%f" % (key, value)
