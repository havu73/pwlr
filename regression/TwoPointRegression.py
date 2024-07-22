from sklearn.linear_model import LinearRegression
import numpy as np


class TwoPointRegression(LinearRegression):
    '''
    a class that inherits the LinearRegression class in sklearn.linear_model, but with the following requirements:
    (1) class name: TwoPointRegression.
    (2) It takes in xs and ys, which each has length of 2 (xs also has only 1 feature).
    (3) If xs are of the same value and ys are of different values, set the class vertical_line=True in the fit function,
    else run the sklearn.linear_model.LinearRegression.fit function.
    (4) for the predict function, it can take is 1D array and reshape it into 2D array, and if the flag vertical_line is True it returns np.nan,
    else runs the function sklearn.linear_model.LinearRegression.predict
    '''
    def __init__(self):
        super().__init__()
        self.vertical_line = False

    def fit(self, xs, ys):
        # Check if xs has length 2 and only 1 feature
        if len(xs) != 2 or len(xs[0]) != 1:
            raise ValueError("xs must have length of 2 and each x must have only 1 feature")

        # Check if ys has length 2
        if len(ys) != 2:
            raise ValueError("ys must have length of 2")

        # Check if xs are of the same value and ys are different
        if xs[0][0] == xs[1][0] and ys[0] != ys[1]:
            self.vertical_line = True
        else:
            super().fit(xs, ys)
            self.vertical_line = False

    def predict(self, X):
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.vertical_line:
            return np.full((len(X),), np.nan)
        else:
            return super().predict(X)


# Example usage
# xs = np.array([[1], [1]])
# ys = np.array([2, 3])
# model = TwoPointRegression()
# model.fit(xs, ys)
# print("Vertical line:", model.vertical_line)  # Should be True
# print("Predictions:", model.predict(np.array([1, 2, 3])))  # Should be [nan, nan, nan]
#
# xs = np.array([[1], [2]])
# ys = np.array([2, 3])
# model.fit(xs, ys)
# print("Vertical line:", model.vertical_line)  # Should be False
# print("Predictions:", model.predict(np.array([1, 2, 3])))  # Should run LinearRegression predict
