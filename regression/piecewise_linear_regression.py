
import numpy as np
SEED=9999
np.random.seed(SEED)
from .TwoPointRegression import TwoPointRegression
from scipy import optimize


class PiecewiseLinearRegression:
    def __init__(self, no_trailing_zeros=True):
        self.trained = False
        self.px = None
        self.py = None
        self.no_trailing_zeros = no_trailing_zeros


    def fit(self, x, y, fixed_breaks=None, max_segments:int=100):
        if self.no_trailing_zeros:
            px, py = piecewise_linearRegression_no_trailing_zeroes(x, y, fixed_breaks=fixed_breaks, max_segments=max_segments)
        else:
            px, py = piecewise_linearRegression(x, y, fixed_breaks=fixed_breaks, max_segments=max_segments)  # function declared outside of class.
        self.px = px
        self.py = py
        self.trained = True

    def save_model(self, filename):
        if not self.trained:
            raise ValueError('Model has not been trained yet')
        np.savez(filename, px=self.px, py=self.py)

    def load_model(self, filename):
        data = np.load(filename)
        self.px = data['px']
        self.py = data['py']
        self.trained = True

    def x_to_y_array(self, xs):
        return x_to_y_array(xs, self.px, self.py)  # ys_filtered, xs_filtered


    def y_to_x_array(self, ys):
        return y_to_x_array(ys, self.px, self.py)


def polish_fixed_breaks(fixed_breaks, xmin, xmax):
    '''
    Given a list of fixed breakpoints, make sure that it is within the range of [xmin, xmax]
    :param fixed_breaks:
    :param xmin:
    :param xmax:
    :return:
    '''
    if fixed_breaks is None:
        return [xmin, xmax]
    fixed_breaks = [x for x in fixed_breaks if (x>=xmin) & (x<=xmax)]
    if xmin < fixed_breaks[0]:
        fixed_breaks = [xmin] + fixed_breaks
    if xmax > fixed_breaks[-1]:
        fixed_breaks = fixed_breaks + [xmax]
    return fixed_breaks


def find_gap(X):
    '''
    Given a list of x-coordinates, find the gap between the x-coordinates (usually, it is either 1 or 0.001 depending on whether we are calculating the coverage in terms of bp or kb)
    But there can be cases when the gaps is variables (bc of subsampling)
    In that case we will return the median gap
    :param X:
    :return:
    '''
    X_diff = np.diff(X)
    if np.all(X_diff == X_diff[0]):
        return X_diff[0]
    else:
        return np.median(X_diff)

def piecewise_linearRegression_no_trailing_zeroes(X, Y, fixed_breaks = None, max_segments:int=100):
    '''
    We will first get rid of the trailing zeroes in the coverage data, and then fit the piece-wise linear regression
    Assumption: Y is the culmulative coverage at each position, so the step of calculating culmulative coverage has already been done
    :param X: position along the gene
    :param Y: coverage at each position
    :param max_segments: max number of segments to fit in piecewise linear regression
    :return:
    '''
    # first, filter out the trailing rows (positions) where BOTH of the timepoints have zero coverage
    last_non_zero_index = np.where(Y != 0)[0][-1]
    X = X[:last_non_zero_index+1]
    Y = Y[:last_non_zero_index+1]
    px, py = piecewise_linearRegression(X, Y, fixed_breaks=fixed_breaks, max_segments=max_segments)
    return px, py

def piecewise_linearRegression(X, Y, fixed_breaks=None, max_segments:int=100):
    """
    Fit the segments of the coverage data to piece-wise linear regression
    I modified this code from https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2e
    the assumption of this code is that:
    X are strictly increasing, and X can be equally spaced or not
    :param X: np px of size (n,) where n is the number of data points
    :param Y: np px of size (n,) where n is the number of data points
    :param max_segments: maximum number of segments to fit. This function will find the optimal number of segments to fit
    :return:
    """
    gapX = find_gap(X)  # find the gap between the x-coordinates. this will be used to determine the fixed breakpoints' y coordinates
    xmin = X.min()
    xmax = X.max()
    n = len(X)
    AIC_ = float('inf')
    BIC_ = float('inf')
    fixed_breaks= polish_fixed_breaks(fixed_breaks, xmin, xmax)
    r_ = None
    for count in range(1, max_segments + 1):
        seg = np.full(count - 1, (xmax - xmin) / count)
        px_init = seg.cumsum()
        py_init = [Y[np.abs(X - x) <= 1].mean() for x in px_init]
        pxy_init = np.r_[px_init, py_init]
        def define_breakpoints(pxy, Y, fixed_breaks):
            fixed_py = [Y[np.abs(X - x) <= gapX].mean() for x in fixed_breaks]
            # first half of pxy is px, second half of pxy is py
            px = pxy[:len(pxy) // 2]
            py = pxy[len(pxy) // 2:]
            px = np.r_[px, fixed_breaks]
            py = np.r_[py, fixed_py]
            sorted_indices = np.argsort(px)
            px_sorted = px[sorted_indices]
            py_sorted = py[sorted_indices]
            return px_sorted, py_sorted
        def err(flex_pxy):  # This is RSS / n
            px, py = define_breakpoints(flex_pxy, Y, fixed_breaks)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)
        r = optimize.minimize(err, x0=pxy_init, method='Nelder-Mead')
        AIC = n * np.log(err(r.x)) + 2 * count
        BIC = n * np.log(err(r.x)) + 1 * count * np.log(n)
        if ((BIC < BIC_) & (AIC < AIC_)):  # Continue adding complexity.
            r_ = r
            AIC_ = AIC
            BIC_ = BIC
    px, py = define_breakpoints(r_.x, Y, fixed_breaks)  # px is always increasing here
    px = minor_fix_px(px) # make sure that it is increasing
    py = minor_fix_py(py) # make sure that it is decreasing
    return px, py


def determine_increase_decrease(px):
    '''
    Given a list of x-coordinates, determine whether the x-coordinates are increasing or decreasing
    :param px:
    :return:
    '''
    # determine whether px is increasing or decreasing
    px_diff = np.diff(px)
    increasing= np.all(px_diff >= 0)
    if increasing:
        pass
    elif np.all(px_diff <=  0): # all decreasing
        increasing = False
        px = px[::-1] # reverse the order of px to make it increasing
    else:
        raise ValueError('function find_segment_index: px should be either increasing or decreasing: px/y: ', px)
    return increasing, px


def search_sorted(px, xs):
    '''
    Given a non-decreasing array px, return the indices of where the values of xs should be inserted into px
    If xs is less than the first element of px, return 0 (the first segment index)
    If xs is greater than the last element of px, return len(px)-1 (the last segment index)
    If two consecutive elements of px are the same, return the index of the first element
    :param px: [0,1,1,2]
    :param xs: 1
    :return: 1
    '''
    if xs < px[0]:
        return 0
    if xs >= px[-1]:
        return len(px)-2
    curr_idx = 0
    while True:
        if px[curr_idx] <= xs < px[curr_idx+1]:
            return curr_idx
        elif px[curr_idx] == xs and px[curr_idx+1] == xs:
            return curr_idx
        curr_idx += 1

def search_sorted_array(px, xs):
    '''
    Given a non-decreasing array px, return the indices of where the values of xs should be inserted into px
    There are specific rules that only apply to our case:
    - If xs is less than the first element of px, return 0 (the first segment index)
    - If xs is greater than the last element of px, return len(px)-2 (the last segment index)
    - If two consecutive elements of px are the same, return the index of the first element
    If px has N elements, then there are N-1 segments between each of the elements of px
    :param px:
    :param xs:
    :return:
    '''
    search_func = np.vectorize(lambda x: search_sorted(px, x))
    return search_func(xs)

def find_segment_index(xs,px):
    """
    Given a value of x, find the segment index that the value of x falls into
    :param x: a np px of x xs that we want to insert into px list. size: (n,)
    :param px: a list of increasing/decreasing x-coordinates --> segments endpoints
    :return:
    """
    if len(px.shape)>1:  # change from (n,1) to (n,)
        px = px.reshape(-1)
    isIncreasing, px_to_sort = determine_increase_decrease(px)
    # np.searchsorted(px, xs) assumes px is increasing. Therefore, if px is decreasing, we need to reverse the search side
    indices = search_sorted_array(px_to_sort, xs)
    # now, if increasing is False, we need to reverse the indices back to the original order
    if not isIncreasing:
        indices = len(px) - 2 - indices
    return indices

def assert_non_decreasing(px):
    """
    Given a list of x-coordinates, assert that the elements in px are non-decreasing
    """
    px_diff = np.diff(px)
    assert np.all(px_diff >= 0), f'px should be non-decreasing px_diff {px_diff}'

def assert_non_increasing(py):
    """
    Given a list of y-coordinates, assert that the elements in py are non-increasing
    """
    py_diff = np.diff(py)
    assert np.all(py_diff <= 0), f'py should be non-increasing: py_diff {py_diff}'

def find_horizontal_line_pxs(px, py):
    """
    Given a list of x-coordinates and y-coordinates that specify the endpoints of the piece-wise linear regression, find the indices of the horizontal lines
    Here, horizontal lines means that px[i] < px[i+1] but py[i] == py[i+1]
    :param px:
    :param py:
    :return: list of (px[i], px[i+1]) values that correspond to horizontal lines
    """
    results = []
    for i in range(len(px)-1):
        if py[i] == py[i+1] and px[i] < px[i+1]:
            results.append((px[i], px[i+1]))
    return results

def idx_xs_out_ranges(xs, pxs_to_filter):
    """
    Given a list of x-coordinates and a list of ranges, filter out the x-coordinates that fall within the ranges
    :param xs: a list of x values
    :param pxs_to_filter: a list of tuples specifying the ranges
    :return: mask: a boolean mask where True indicates that the x-coordinate SHOULD BE INCLUDED (not within the range)
    """
    # Create a boolean mask initialized to False
    mask = np.zeros(xs.shape, dtype=bool)
    # Update the mask for each range
    for start, end in pxs_to_filter:
        mask |= (xs >= start) & (xs <= end)  # True means that the x-coordinate should be filtered out bc it is within the range
    return ~mask  # we want to flip the bit, so True means it should be included

def filterOut_horizontal_lines(xs, ys, px, py):
    '''
    Piece-wise linear regression is specified in px, py
    xs and ys are known points on this piece-wise linear regression (previously calculated)
    We want to filter out the points that lie on the horizontal line
    :param xs: x-coordinates
    :param ys: correpsponding y-coordinates based on px, py piece-wise linear regression
    :param px: a list of x-coordinates of endpoints in piece-wise linear regression
    :param py: a list of y-coordinates of endpoints in piece-wise linear regression
    :return:
    '''
    pxs_horizontal = find_horizontal_line_pxs(px, py)  # list of (px[i], px[i+1]) values that correspond to those horizontal lines
    mask = idx_xs_out_ranges(xs, pxs_horizontal)
    return xs[mask], ys[mask]

def x_to_y_array(xs, px, py):
    """
    Given a value of x, find the corresponding y-coordinate
    We cannot use the default np.interp because itwould not consider the case where x is outside the range of px
    If x is before the first element of px, we will use the first segment's model
    If x is after the last element of px, we will use the last segment's model
    Given xs and the endpoints of the segments of piece-wise linear regression specified by px and py, calculate the y-coordinates
    Assumptions: px is non-decreasing, py is non-increasing
    px and py specify a mathematical function (for each value of x there is exactly one value of y)
    An added functionality of this function compared to what is normally implemented in regular linear regression is that
    if there is a horizontal line (py[i] == py[i+1] but px[i] <px[i+1]),
    then should cut both the x and y coordinates at that point out of the output
    :param xs: np.array of x-coordinates
    :param px: a list of x-coordinates
    :param py: a list of y-coordinates. Length of px and py should be the same. Each elements show the x and y coordinates of the endpoints of piece-wise linear regression
    :return: xs, ys: xs should get rid of the x-coordinates that are on the horizontal line, and ys should get rid of the corresponding y-coordinates
    """
    xs = np.array(xs)
    assert_non_decreasing(px)  # assert px is non decreasing
    assert_non_increasing(py)  # assert py is non increasing
    px_to_model = px.reshape(-1,1) # needed because our data is single-feature
    num_model= px.shape[0]-1
    model_list = list(map(lambda i: TwoPointRegression(), range(num_model)))
    for i in range(num_model):
        model_list[i].fit(px_to_model[i:i+2], py[i:i+2])
    segment_indices = find_segment_index(xs,px) # list of segment indices that each x falls into (-1 if x is outside the range of px)
    xs_to_model = xs.reshape(-1,1) # needed because our data is single-feature
    N = xs.shape[0]
    ys = list(map(lambda i: (model_list[segment_indices[i]]).predict(xs_to_model[i:(i+1)])[0], range(N)))
    # the last step: filter out the xs and ys values that correspond to points lying on the horizontal line in the piece-wise linear regression
    xs_filtered, ys_filtered = filterOut_horizontal_lines(xs, np.array(ys), px, py)
    return ys_filtered.squeeze(), xs_filtered.squeeze()

def y_to_x_array(ys, px, py):
    """
    Given a value of y, find the corresponding x-coordinate
    :param ys: np.array of y-coordinates
    :param px: a list of x-coordinates
    :param py: a list of y-coordinates. Length of px and py should be the same. Each elements show the x and y coordinates of the endpoints of piece-wise linear regression
    :return:
    """
    ys = np.array(ys)
    assert_non_decreasing(px)  # assert px is non decreasing
    assert_non_increasing(py)  # assert py is non increasing
    # note since px, py are np px, indexing system is set [include,exclude)
    py=py.reshape(-1,1) # needed because our data is single-feature
    num_model= py.shape[0]-1
    model_list = list(map(lambda i: TwoPointRegression(), range(num_model)))
    for i in range(num_model):
        model_list[i].fit(py[i:i+2], px[i:i+2])
    segment_indices = find_segment_index(ys,py) # list of segment indices that each y falls into (-1 if y is outside the range of py)
    ys = ys.reshape(-1,1) # needed because our data is single-feature
    N = ys.shape[0]
    xs = list(map(lambda i: (model_list[segment_indices[i]]).predict(ys[i:(i+1)])[0], range(N)))
    return np.array(xs).squeeze()  # squeeze to convert from (N,1) to (N,)

def minor_fix_py(py):
    """
    Given a list of y-coordinates, make sure that the entries are decreasing
    49.91352062 50.1785993  49.76237748 47.21655337 --> 49.91352062 49.91352062 49.76237748 47.21655337
    """
    if len(py)<=1:
        return py
    curr_y = py[0]
    for i in range(1,len(py)):
        if py[i] > curr_y:
            print('fixed. py[i] - curr_y: ', py[i]-curr_y)
            print('py[i]: ', py[i])
            print('curr_y: ', curr_y)
            py[i] = curr_y
        else:
            curr_y = py[i]
    return py

def minor_fix_px(px):
    """
    Given a list of x-coordinates, make sure that the entries are increasing
    """
    if len(px)<=1:
        return px
    curr_x = px[0]
    for i in range(1,len(px)):
        if px[i] < curr_x:
            print('fixed. px[i] - curr_x: ', px[i]-curr_x)
            print('px[i]: ', px[i])
            print('curr_x: ', curr_x)
            px[i] = curr_x
        else:
            curr_x = px[i]
    return px

# # Generate x xs
# x = np.linspace(0, 15, 300)
#
# # Calculate y xs based on x xs with different conditions and added noise
# y = np.piecewise(x, [x < 5, (x >= 5) & (x < 10), x >= 10],
#                  [lambda x: -x + 10 + np.random.normal(0, 0.5, size=x.shape),
#                   lambda x: -0.5 * x + 6 + np.random.normal(0, 0.5, size=x.shape),
#                   lambda x: -0.25 * x +4+ np.random.normal(0, 0.5, size=x.shape)])
#
# px, py = piecewise_linearRegression(x, y, fixed_breaks=[], max_segments=15)
# print('first, px: ', px)
# print('first, py: ', py)
# px, py = piecewise_linearRegression(x, y, fixed_breaks=[5, 10], max_segments=15)
# print('second, px: ', px)
# print('second, py: ', py)
