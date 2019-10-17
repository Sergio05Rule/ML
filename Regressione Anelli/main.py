import pandas as pd
from Regression.utils import *

data = pd.read_csv("wine.csv", sep=',', header=None)
data = data.as_matrix()
X = data[:, 0:data.shape[1]-1]
y = data[:, -1]

print(X[:, 9])
# plotData(X[:, 9], y)

# normalization
mu, sigma = muSigma(X)
X = zScore(X, mu, sigma)

print(np.ones(data.shape[0]).T)
X = np.column_stack((np.ones(data.shape[0]), X))

theta = np.zeros(X.shape[1])
print(theta)

#hyperparameters
alpha = 0.6
num_iters = 5

print("Cost at iteration 0: ", computeCost(X, y, theta))

#Gradient Descent
theta, history = gradientDescent(X, y, theta, alpha, num_iters)
#theta = normalEquations(X, y)
#plotLearning(history)

new_tuple = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]

new_tuple = zScore(new_tuple, mu, sigma)
new_tuple = np.insert(new_tuple, 0, 1)

print(predict(new_tuple, theta))


