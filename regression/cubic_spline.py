from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import numpy as np
SEED=9999
np.random.seed(SEED)

def piecewise_linearRegression(X, Y, maxcount:int=100):
    """
    Fit the segments of the coverage data to piece-wise linear regression
    I copied this code from https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2e
    :param X: np array of size (n,) where n is the number of data points
    :param Y: np array of size (n,) where n is the number of data points
    :param maxcount: maximum number of segments to fit. This function will find the optimal number of segments to fit
    :return:
    """
    return PchipInterpolator(X, Y)  # this is a function that is returned. It can be used like the following: f = piecewise_linearRegression(X, Y); f(x) will return the interpolated value of y at x

def x_to_y_array(xs, xy_spline):
    """
    Given a value of x, find the corresponding y-coordinate
    :param xs: np.array of x-coordinates
    :param px: a list of x-coordinates
    :param py: a list of y-coordinates. Length of px and py should be the same. Each elements show the x and y coordinates of the endpoints of piece-wise linear regression
    :return:
    """
    return xy_spline(xs)
