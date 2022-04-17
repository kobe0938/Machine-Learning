import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n, _ = x.shape
    e = np.random.choice(K, n)
    mu = [np.mean(x[e == k, :], axis = 0) for k in range(K)]
    sigma = [np.cov(x[e == k, :].T) for k in range(K)]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K,), fill_value=(1. /K), dtype=np.float32)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), fill_value=(1. /K), dtype=np.float32)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        w = E_step(x, w, phi, mu, sigma)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = M_step(x, w, mu, sigma)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = logllh(x, phi, mu, sigma)
        it = it + 1
        print('[iter: {:03d}. log_llh: {: .4f}]'.format(it, ll))
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w = E_step(x, w, phi, mu, sigma)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = M_step_ss(x, x_tilde, z_tilde, w, phi, mu, sigma, alpha)


        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = logllh(x, phi, mu, sigma)
        ll = ll + alpha * logllh(x_tilde, phi, mu, sigma, z_tilde)
        it = it + 1
        print('[iter: {:03d}. log_llh: {: .4f}]'.format(it, ll))
        # *** END CODE HERE ***
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Define any helper functions
def E_step(x, w, phi, mu, sigma): 
    length_of_mu = len(mu)
    row, _ = x.shape

    for m in range(row):
        for n in range(length_of_mu):
            w[m,n] = x_given_z(x[m], mu[n], sigma[n]) * phi[n]
    
    w = w/np.sum(w, axis=1, keepdims=True)
    return w

def M_step(x, w, mu, sigma):
    length_of_mu = len(mu)
    row, _ = x.shape
    phi = np.mean(w, axis = 0)
    for m in range(length_of_mu):
        wj = w[:, m: m + 1]
        #note here
        mu[m] = np.sum(wj * x, axis = 0) / np.sum(wj)
        sigma[m] = np.zeros_like(sigma[m])
        for n in range(row):
            tmp = x[n] - mu[m]
            sigma[m] = sigma[m] + w[n, m] * np.outer(tmp, tmp)
        sigma[m] = sigma[m] / np.sum(wj)
    return phi, mu, sigma

def logllh(x, phi, mu, sigma, z=None):
    length_of_mu = len(phi)
    row, _ = x.shape
    ll = 0.
    for m in range(row):
        if z is None:
            px = 0.
            for n in range(length_of_mu):
                px = px + x_given_z(x[m], mu[n], sigma[n]) * phi[n]
        else:
            px = x_given_z(x[m], mu[int(z[m])], sigma[int(z[m])] * phi[int(z[m])])
        ll = ll + np.log(px)
    return ll

def x_given_z(x, mu, sigma):
    # TODO: probably not needed
    length_of_x = len(x)
    assert length_of_x == len(mu) and sigma.shape == (length_of_x, length_of_x), 'error: not match demension'
    if (np.linalg.det(sigma) != 0):
        q = 1./((2. * np.pi) ** (length_of_x/2) * np.sqrt(np.linalg.det(sigma)))
        sigma_invert = np.linalg.inv(sigma)
        x_mu = x - mu
        p = q * np.exp(-0.5 * (x_mu).dot(sigma_invert).dot(x_mu.T))
        return p
    return 0
def M_step_ss(x, x_, z, w, phi, mu, sigma, a):
    length_of_mu = len(mu)
    row_, _ = x_.shape
    row, _ = x.shape
    w_sum = np.sum(w, axis=0)
    k_count = [np.sum(z == o) for o in range(length_of_mu)]
    for i in range(length_of_mu):
        phi[i] = (w_sum[i] + a * k_count[i])/(row + a * row_)
        wj = w[:,i:i+1]
        mu[i] = ((np.sum(wj * x, axis = 0)) + a * np.sum(x_[(z == i).squeeze(), :], axis=0))/(np.sum(wj) + a * k_count[i])
        sigma[i] = np.zeros_like(sigma[i])
        for j in range(row):
            sigma[i] = sigma[i] + w[j, i] * np.outer(x[j] - mu[i], x[j] - mu[i])
        for j in range(row_):
            if z[j] == i:
                sigma[i] = sigma[i] + a * np.outer(x_[j] - mu[i], x_[j] - mu[i])
        sigma[i] = sigma[i]/(np.sum(wj) + a * k_count[i])

    return phi, mu, sigma

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
