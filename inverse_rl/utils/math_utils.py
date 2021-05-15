import numpy as np


def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov = np.square(np.exp(log_diag_std))
    diff = x - mean
    exp_term = -0.5 * np.sum(np.square(diff) / cov, axis=1)
    norm_term = -0.5 * d * np.log(2 * np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs


def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))
