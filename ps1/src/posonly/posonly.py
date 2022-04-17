import numpy as np
import util
import sys

sys.path.append('/Users/chenxiaokun/Desktop/cs229/ps1/src/linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path,label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path,label_col='t', add_intercept=True)
    case_true = LogisticRegression()
    case_true.fit(x_train, t_train)
    util.plot(x_test, t_test, case_true.theta, '/Users/chenxiaokun/Desktop/cs229/ps1/src/posonly/q2a.png')
    case_true_pred = case_true.predict(x_test)
    np.savetxt(output_path_true, case_true_pred, fmt='%10.5f')

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, n_train = util.load_dataset(train_path,label_col='y', add_intercept=True)
    x_test, n_test = util.load_dataset(test_path,label_col='y', add_intercept=True)
    case_naive = LogisticRegression()
    case_naive.fit(x_train, n_train)
    util.plot(x_test, n_test, case_naive.theta, '/Users/chenxiaokun/Desktop/cs229/ps1/src/posonly/q2b.png')
    case_naive_pred = case_naive.predict(x_test)
    np.savetxt(output_path_naive, case_naive_pred, fmt='%10.5f')
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    x_valid, y_valid = util.load_dataset(valid_path,label_col='y', add_intercept=True)
    x_valid = x_valid[y_valid == 1, :]
    e_pred = case_naive.predict(x_valid)
    a = np.mean(e_pred)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    # correlation = 1 + (np.log(2/np.mean(case_naive.predict(x_valid)) - 1) / case_naive.theta[0])
    util.plot(x_test, t_test, case_naive.theta,'/Users/chenxiaokun/Desktop/cs229/ps1/src/posonly/q2f.png', a)
    np.savetxt(output_path_adjusted, case_naive_pred/a, fmt='%10.5f')
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
