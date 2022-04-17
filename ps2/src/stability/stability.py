# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0

    for j in range(100):
        if Y[j] == 1.0:
            plt.scatter(X[j,1],X[j,2], color = 'g',s = 100)
        else:
            plt.scatter(X[j,1],X[j,2], color = 'r',s = 100)
    plt.xticks(rotation = 25)
    plt.xlabel('Names')
    plt.ylabel('Values')
    plt.title('Patients Blood Pressure Report', fontsize = 20)
    
    plt.show()

    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
