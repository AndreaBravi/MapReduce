from nose.tools import eq_ as assert_equal
from numpy.testing.utils import assert_almost_equal, assert_array_equal
import numpy as np
from StringIO import StringIO
from shopify.utilities import MRData, MRLogisticRegression

INF = np.Inf

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

def test_MRData_process():
    data = MRData()

    line = '0,0.7,20\n1,0.3,40\n1,0.5,20'

    X, y = data.process(StringIO(line))

    expected_result = np.array([[ 1.22474487, -0.70710678],
                                [-1.22474487,  1.41421356],
                                [ 0.        , -0.70710678]])

    assert_almost_equal(X, expected_result, decimal=7)
    assert_array_equal(y, np.array([0, 1, 1], ndmin=2).T)


def test_MRData_clean_nans():
    data = np.array([[1, 2, 3, np.nan], [np.nan, 4, 5, 6]])

    result = MRData()._clean_nans(data)
    expected_result = np.array([[1, 2, 3, 6], [1, 4, 5, 6]])

    assert_array_equal(result, expected_result)

def test_MRData_normalize():
    data = np.array([[1, 2, 3], [4, 5, 6], [11, 32, 44]], dtype='float')

    result = MRData()._normalize(data)
    expected_result = np.array([[-1.03422447, -0.81537425, -0.78596495],
                                [-0.31822291, -0.59299945, -0.62519939],
                                [ 1.35244738,  1.4083737 ,  1.41116435]])

    assert_almost_equal(result, expected_result, decimal=7)



def test_MRLogisticRegression_gradient():
    X = np.array([[1, 0],
                  [0, 1]], dtype='float')
    y = np.array([0, 1], dtype=float, ndmin=2).T
    b = np.zeros([2, 1])

    result = MRLogisticRegression()._gradient(X, y, b)

    expected_result = np.array([-0.5, 0.5], ndmin=2).T
    assert_array_equal(result, expected_result)
    return result

def test_MRLogisticRegression_hessian():
    X = np.array([[1, 0],
                  [0, 1]], dtype='float')
    y = np.array([0, 1], dtype=float, ndmin=2).T
    b = np.zeros([2, 1])

    result = MRLogisticRegression()._hessian(X, y, b)

    expected_result = np.array([[-0.5, 0],[0, -0.5]])

    assert_array_equal(result, expected_result)
    return result

def test_MRLogisticRegression_mapper():
    X = np.array([[1, 0],
                  [0, 1]], dtype='float')
    y = np.array([0, 1], dtype=float, ndmin=2).T

    g_ok = test_MRLogisticRegression_gradient()
    H_ok = test_MRLogisticRegression_hessian()

    g, H = MRLogisticRegression().mapper(X, y)

    assert_array_equal(g, g_ok)
    assert_array_equal(H, H_ok)


def test_MRLogisticRegression_reducer():

    g_ok = test_MRLogisticRegression_gradient()
    H_ok = test_MRLogisticRegression_hessian()

    expected_result = np.array([-1, 1], dtype=float, ndmin=2).T

    result = MRLogisticRegression().reducer([(g_ok, H_ok)])

    assert_array_equal(result, expected_result)

