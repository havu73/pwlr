import numpy as np
SEED=9999
np.random.seed(SEED)
import torch
import torch.nn as nn
import torch.optim as optim
def load_model(fn):
    '''
    Load the model from a file
    :param fn:
    :return:
    '''
    model = torch.load(fn)
    return model

def determine_increase_decrease(px):
    '''
    Given a list of x-coordinates, determine whether the x-coordinates are increasing or decreasing
    :param px:
    :return:
    '''
    # determine whether px is increasing or decreasing
    px_diff = torch.diff(px)
    increasing= torch.all(px_diff >= 0)
    if increasing:
        pass
    elif torch.all(px_diff<=0): # all decreasing
        increasing = False
        px = torch.flip(px,(0,)) # reverse the order of px to make it increasing
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
    if np.isnan(xs):
        return -1
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
    vectorized_function = np.vectorize(lambda t: search_sorted(px, t))
    idx = vectorized_function(xs.detach().numpy())
    idx = torch.tensor(idx, dtype=torch.float)
    # idx = torch.tensor([search_sorted(px, x) for x in xs.detach().numpy()], dtype=torch.float)
    return idx

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
    indices = search_sorted_array(px_to_sort, xs)  # this function is designed such that if xs is nan, then the segment index is nan
    # now, if increasing is False, we need to reverse the indices back to the original order
    if not isIncreasing:
        indices[indices == -1] = torch.nan
        indices = len(px) - 2 - indices
        # convert back nan values to -1
        indices[torch.isnan(indices)] = -1  ## this is because if we have nan then the indices cannot be converted to int
    indices = torch.tensor(indices, dtype=torch.long)
    return indices

def convert_to_torch(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return x


def find_increase_ranges(tensor):
    '''
    Given a tensor of size (N,), find ranges within the tensor where the values are increasing instead of decreasing
    This function is used to find, in py, if there are ranges of Y that we need to filter out
    :param tensor:
    :return:
    '''
    # Initialize an empty list to store the ranges
    ranges = []
    # Traverse the tensor to find ranges where the values decrease and then increase
    start = None
    end = None
    downhill=True
    for i in range(1, len(tensor)):
        if (tensor[i] > tensor[i - 1]) and downhill:  # the sequence start to increase
            downhill=False
            start = tensor[i - 1]
        elif (tensor[i] > tensor[i - 1]) and not downhill:  # it keeps increasing
            pass
        elif (tensor[i] <= tensor[i - 1]) and not downhill:  # the sequence start to decrease
            end = tensor[i - 1]
            ranges.append((start, end))
            downhill=True
        else: # tensor[i] <= tensor[i - 1] and downhill=True --> keep decreasing, let it be
            pass
    # we have reached the end of the tensor
    if not downhill: # we are going uphill and reach the end of the tensor
        end = tensor[-1]
        ranges.append((start, end))
    else: # we are going downhill, which means the uphill past has been recorded
        pass
    return ranges

# Define the piecewise linear model
class PiecewiseLinearRegression(nn.Module):
    def __init__(self, px=None, gap=0.1, x_min=0, x_max=10, y_decrease_hor=0):
        super(PiecewiseLinearRegression, self).__init__()
        print('Note that the PiecewiseLinearRegression is designed to work with the very ultimate assumption that the fragments are fixed in the x-axis')
        if px is None:
            self.xgap = gap
            px = np.arange(x_min, x_max+gap, gap).round(1)
            px = torch.tensor(px, dtype=torch.float32)
        self.px = px
        self.xgap = px[1] - px[0]
        # assert that the x-coordinates are evenly spaced, to certain precision
        assert torch.allclose(torch.diff(px), self.xgap, atol=1e-04), 'The x-coordinates should be evenly spaced'
        self.py = nn.Parameter(torch.zeros(len(px), dtype=torch.float32))
        self.y_decrease_hor = y_decrease_hor  # delta_y such that if slope >= -delta_y/xgap, then the slope is considered to be horizontal
        self.trained = False
        self.inverted= False
    def forward(self, x):
        # get the index of the models that x belongs to, should be integer
        idx = (x / self.xgap).long()  # this will round down
        # if idx is >= len(px), then we will set it to len(px)-1
        idx = torch.clamp(idx, 0, len(self.px) - 2)
        slope = (self.py[idx + 1] - self.py[idx]) / self.xgap
        intercept = self.py[idx] - slope * self.px[idx]
        result = slope * x + intercept
        return result
    def fit(self, X, Y, criterion=nn.MSELoss(), optimizer=None, num_epochs=5000):
        '''

        :param X:
        :param Y:
        :param criterion:
        :param optimizer:
        :param num_epochs:
        :param y_hor_thres: slope = delta_y/delta_x, where delta_x=self.xgap. If delta_y < y_hor_thres, then the slope is considered to be horizontal
        :return:
        '''
        X = convert_to_torch(X)
        Y = convert_to_torch(Y)
        optimizer = optim.Adam(self.parameters(), lr=0.01) if optimizer is None else optimizer
        # initialize the py values to be close to the Y values
        with torch.no_grad():  # initialize the py values to be close to the observed Y values
            self.py.copy_(torch.tensor([Y[np.abs(X - x) <= 1].mean() for x in self.px], dtype=torch.float))
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X)  # given X, predict Y
            loss = criterion(outputs, Y)  # minimize the MSE loss between the predicted Y and the true Y
            # add loss to make sure that the py is decreasing
            for i in range(1, len(self.py)):
                loss += torch.max(torch.tensor(0, dtype=torch.float32), self.py[i] - self.py[i - 1])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        self.trained = True
        self.slope = (self.py[1:] - self.py[:-1]) / self.xgap
        self.intercept = self.py[:-1] - self.slope * self.px[:-1]
        # slope = delta_y/delta_x, where delta_x=self.xgap.
        # assumption: Y is decreasing --> slope is negative. If the decreasing in y is less than y_decrease_hor for each unit of self.xgap, then the slope is considered to be horizontal.
        # in other words, if slope >= -y_decrease_hor/xgap, then the slope is considered to be horizontal
        self.horizontal_lines = self.slope >= (-self.y_decrease_hor/self.xgap)
        return
    def print_model(self):
        print("Optimized py values:", self.py.data)
        print("Optimized px values:", self.px)
        return
    def save_model(self, fn):
        '''
        Save the model (px and py) into a file
        :param fn:
        :return:
        '''
        if self.trained:
            torch.save(self, fn)
        else:
            print('Model has not been trained yet. Cannot save the model.')
        return
    def x_to_y(self, x):
        if not self.trained:
            print('Model has not been trained yet. Cannot predict y values.')
            return
        self.eval()
        x = convert_to_torch(x)
        x_idx = (x/ self.xgap).long()
        x_idx = torch.clamp(x_idx, 0, len(self.px) - 2)
        # filter OUT regions that correspond to a horizontal line (slope = 0)
        # for each x, get true/false if the corresponding x_idx is among the horizontal lines
        on_horizontal = self.horizontal_lines[x_idx]
        # filter out the x for which the corresponding x_idx is among filter_idx
        x_filtered = x[~on_horizontal]
        y_filtered = self(x_filtered)
        return x_filtered, y_filtered
    def inverse_lines(self):
        if not self.trained:
            print('Model has not been trained yet. Cannot predict x values.')
            return
        if self.inverted:
            return
        self.eval()
        self.inverse_slope = 1 / self.slope
        # if it is a horizontal line, then the inverse slope is nan (instead of infinity because we want it to signal to use that the point is invalid)
        self.inverse_slope[self.horizontal_lines] = torch.nan
        self.inverse_intercept = -self.intercept / self.slope
        # add one nan at the end of inverse_slope and inverse_intercept to be used in case we do not have a valid segment index
        self.inverse_slope = torch.cat((self.inverse_slope, torch.tensor([torch.nan], dtype=torch.float32)))
        self.inverse_intercept = torch.cat((self.inverse_intercept, torch.tensor([torch.nan], dtype=torch.float32)))
        self.inverted = True
        return
    def _filter_y_in_increase_ranges(self, y):
        '''
        Filter out the values of y that belong to ranges where self.py is increasing. Those are invalid values. Those values within y will be nan
        :param y:
        :return: y_filtered of length (N,), such that invalid values are nan
        '''
        increase_ranges = find_increase_ranges(self.py.data)
        y_filtered = y.clone()
        for start, end in increase_ranges:
            y_filtered[(y >= start) & (y <= end)] = torch.nan
        return y_filtered
    def _minor_fix_py(self):
        '''
        Fix the py values to make sure that they are decreasing
        :return:
        '''
        # make a copy of self.py
        py_non_increasing = self.py.data.clone()
        for i in range(1, len(self.py)):
            if self.py[i] > py_non_increasing[i - 1]:  # compare the entries with what is currently in the updated py_non_increasing
                py_non_increasing[i] = py_non_increasing[i - 1]
        return py_non_increasing
    def y_to_x(self, y, nan_filter=False):
        if not self.trained:
            print('Model has not been trained yet. Cannot predict x values.')
            return
        self.eval()
        y = convert_to_torch(y)
        # filter out the values of y that belong to ranges where self.py is increasing. Those are invalid values. Those values within y will be nan
        y_filtered = self._filter_y_in_increase_ranges(y)  # y_filtered of length (N,), such that invalid values are nan
        py_non_increasing = self._minor_fix_py()
        idx = find_segment_index(y_filtered, py_non_increasing)
        # convert all the idx that is nan to len(self.py) (because we will have the corresponding nan in self.inverse_slope and self.inverse_intercept)
        idx[torch.isnan(y)] = len(self.slope)
        idx[idx == -1] = len(self.slope)
        self.inverse_lines()
        # assert that length of self.inverse_slope and self.inverse_intercept is equal to len(self.py)+1
        assert len(self.inverse_slope) == len(self.slope)+1, 'Length of inverse_slope should be equal to len(py)+1'
        assert len(self.inverse_intercept) == len(self.intercept)+1, 'Length of inverse_intercept should be equal to len(py)+1'
        assert len(self.py) == len(self.px), 'Length of py should be equal to len(px)'
        assert len(self.inverse_slope) == len(self.inverse_intercept), 'Length of inverse_slope should be equal to length of inverse_intercept'
        x = self.inverse_slope[idx] * y + self.inverse_intercept[idx]  # x is automatically nan if the slope is nan (when the line on x-y plane is horizontal)
        if nan_filter:
            nonNan_filter = ~torch.isnan(x)
            x_filtered = x[nonNan_filter]
            y_filtered = y[nonNan_filter]
            return x_filtered, y_filtered
        return x, y
    def draw_regression_results(self, X, Y=None, save_fn=None):
        '''
        Draw the regression results.
        This plot is to verify that if the model is fit correctly to the data (we do not have data points here), then the x_to_y and y_to_x functions are correct)
        :param X:
        :param Y:
        :return:
        '''
        if not self.trained:
            print('Model has not been trained yet. Cannot draw the regression results.')
            return
        self.eval()
        X = convert_to_torch(X)
        x1, y1 = self.x_to_y(X)
        if Y is None:
            Y = self(X)
        x2, y2 = self.y_to_x(Y, nan_filter=True)
        # plot x and y as points
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(self.px.detach().numpy(), self.py.data.detach().numpy(), label='fit', color='red', alpha=0.3)
        plt.scatter(x2.detach().numpy(), y2.detach().numpy(), label='y_to_x', color='green', s=7, alpha=0.5)
        plt.scatter(x1.detach().numpy(), y1.detach().numpy(), label='x_to_y', color='m', s=7, alpha=0.2)
        plt.legend()
        import os
        if save_fn is not None and os.path.exists(os.path.dirname(save_fn)):
            print('Saving figure to:', save_fn)
            plt.savefig(save_fn)
        return


# import pandas as pd
# fn = '/gladstone/engelhardt/home/hvu/source/RNA_rates/splicingrates/simulations/tests/coverage_df.csv.gz'
# df = pd.read_csv(fn, header = 0, index_col=None, sep = '\t')
# print(df.head())
# tdf = (df).copy()
# tdf.reset_index(drop=True, inplace=True)
# X = tdf.loc[:6480]['position'].values
# Y = tdf.loc[:6480]['0'].values
# X = X/1000
#
# X = torch.tensor(X, dtype=torch.float32)
# Y = torch.tensor(Y, dtype=torch.float32)
# xgap=0.1
# # px should evenly spaced from X.min() to X.max(), with xgap
# px = torch.arange(X.min(), X.max()+xgap, xgap)
# import time
# start = time.time()
# model = PiecewiseLinearRegression(px=px)
# model.fit(X, Y)
# end = time.time()
# print('Time elapsed:', end-start)
# model.print_model()
# x1, y1 = model.x_to_y(X)
# import matplotlib.pyplot as plt
# y2 = torch.arange(Y.min(), Y.max(), 1)
# x2, y2 = model.y_to_x(y2)
# # plot x and y as points
# plt.clf()
# plt.plot(X, Y, label='data', color='blue', alpha=0.3)
# plt.plot(model.px.detach().numpy(), model.py.data.detach().numpy(), label='fit', color='red', alpha=0.3)
# plt.scatter(x2.detach().numpy(), y2.detach().numpy(), label='y_to_x', color='green', s=7, alpha=0.3)
# plt.scatter(x1.detach().numpy(), y1.detach().numpy(), label='x_to_y', color='m', s=7, alpha=0.3)
# plt.legend()
# plt.show()
# plt.savefig('test_pwlr_torch.png')
# model.save_model('test_pwlr_torch.pth')
# model = torch.load(fn)
# y_draw = torch.arange(Y.min(), Y.max(), 1)
# model.draw_regression_results(X, y_draw, save_fn = '/gladstone/engelhardt/home/hvu/source/RNA_rates/splicingrates/simulations/elong_analysis/test_pwlr_regress.png')
