#!/usr/lib/python2.6 python

import sys
from base64 import b64encode as encode
from pickle import dumps
from utilities import MRLogisticRegression, MRData

model = MRLogisticRegression()

data = MRData(sys.stdin)

g, H = model.mapper(data.X, data.y)

key = encode(dumps(data.X[0, :]))
value = encode(dumps((g, H)))

print >> sys.stdout, "%s\t%s" % (key, value)
