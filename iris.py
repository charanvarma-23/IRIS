import numpy as np
import matplotlib.pyplot as plt


class GradientDescent():
    def __init__(self, alpha=0.1, tolerance=0.02, max_iterations=500):
        # alpha is the learning rate or size of step to take in
        # the gradient decent
        self._alpha = alpha
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        # thetas is the array coeffcients for each term
        # the y-intercept is the last element
        self._thetas = None

    def fit(self, xs, ys):
        num_examples, num_features = np.shape(xs)
        self._thetas = np.ones(num_features)

        xs_transposed = xs.transpose()
        for i in range(self._max_iterations):
            # difference between our hypothesis and actual values
            diffs = np.dot(xs, self._thetas) - ys
            # sum of the squares
            cost = np.sum(diffs ** 2) / (2 * num_examples)
            # calculate averge gradient for every example
            gradient = np.dot(xs_transposed, diffs) / num_examples
            # update the coeffcients
            self._thetas = self._thetas - self._alpha * gradient

            # check if fit is "good enough"
            if cost < self._tolerance:
                return self._thetas

        return self._thetas

    def predict(self, x):
        return np.dot(x, self._thetas)


# load some example data
data = np.loadtxt("iris.data.txt", usecols=(0, 1, 2, 3), delimiter=',')
col_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

data_map = dict(zip(col_names, data.transpose()))

# create martix of features
features = np.column_stack((data_map['petal length'], np.ones(len(data_map['petal length']))))

gd = GradientDescent(tolerance=0.022)
thetas = gd.fit(features, data_map['petal width'])
gradient, intercept = thetas

# predict values accroding to our model
ys = gd.predict(features)

plt.scatter(data_map['petal length'], data_map['petal width'])
plt.plot(data_map['petal length'], ys)
plt.show()