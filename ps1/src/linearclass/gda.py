import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    test = GDA()
    test.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, test.theta, '/Users/chenxiaokun/Desktop/cs229/ps1/src/linearclass/q1e.png')
    p_predict = test.predict(x_valid)
    np.savetxt(save_path, p_predict, fmt='%10.5f')
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        row, col = x.shape
        self.theta = np.zeros(col + 1)
        pi = np.sum(y == 1) / row
        miu_0 = np.sum(x[y == 0], axis=0) / (row - sum(y == 1))
        miu_1 = np.sum(x[y == 1], axis=0) / sum(y == 1)
        sig = ((x[y == 0] - miu_0).T.dot(x[y == 0] - miu_0) + (x[y == 1] - miu_1).T.dot(x[y == 1] - miu_1)) / row
        self.theta[0] = 1 / 2 * (miu_0 + miu_1).dot(np.linalg.inv(sig)).dot(miu_0 - miu_1) - np.log((1 - pi) / pi)
        self.theta[1:] = np.linalg.inv(sig).dot(miu_1 - miu_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(np.dot(-x, self.theta)))
        # *** END CODE HERE

if __name__ == '__main__':

    # main(train_path='ds1_train.csv',
    #      valid_path='ds1_valid.csv',
    #      save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
