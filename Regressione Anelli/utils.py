import numpy as np
import matplotlib.pyplot

def predict(X, theta):
    return np.dot(X, theta)

def muSigma(X):
    return (np.mean(X,axis=0), np.std(X,axis=0))

def zScore(X, mu, sigma):
    return np.divide((X - mu), sigma)

def computeCost(X, y, theta):
    m = len(y)
    h = predict(X, theta)
    diff = h - y
    J = (np.sum(diff ** 2)) / (2 * m)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    history = np.zeros(num_iters)

    for i in range(num_iters):
        h = predict(X, theta)
        diff = h - y
        theta = theta - (alpha / m) * np.dot(X.T, diff)
        history[i] = computeCost(X, y, theta)
        print("Cost Function: ", history[i])
    return (theta, history)