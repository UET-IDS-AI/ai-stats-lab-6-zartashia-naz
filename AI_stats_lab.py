import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    if not (0 < theta < 1):
        raise ValueError("Theta must be in (0,1)")

    for x in data:
        if x not in [0, 1]:
            raise ValueError("Data must contain only 0 and 1")

    data = np.array(data)

    # Log-likelihood
    log_likelihood = np.sum(
        data * np.log(theta) + (1 - data) * np.log(1 - theta)
    )

    return float(log_likelihood)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    for x in data:
        if x not in [0, 1]:
            raise ValueError("Data must contain only 0 and 1")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    data = np.array(data)

    # Counts
    num_successes = int(np.sum(data))
    num_failures = int(len(data) - num_successes)

    # MLE
    mle = num_successes / len(data)

    # Log-likelihoods
    log_likelihoods = {}
    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(data, theta)
        log_likelihoods[theta] = ll

    # Best candidate
    best_candidate = None
    best_value = -float("inf")

    for theta, ll in log_likelihoods.items():
        if ll > best_value:
            best_value = ll
            best_candidate = theta

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }


def poisson_log_likelihood(data, lam):
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    if lam <= 0:
        raise ValueError("Lambda must be > 0")

    for x in data:
        if not (isinstance(x, (int, np.integer)) and x >= 0):
            raise ValueError("Data must contain nonnegative integers")

    data = np.array(data)

    # Log-likelihood
    log_likelihood = np.sum(
        data * np.log(lam) - lam - np.array([math.lgamma(x + 1) for x in data])
    )

    return float(log_likelihood)


def poisson_mle_analysis(data, candidate_lambdas=None):
    # Validation
    if len(data) == 0:
        raise ValueError("Data must not be empty")

    for x in data:
        if not (isinstance(x, (int, np.integer)) and x >= 0):
            raise ValueError("Data must contain nonnegative integers")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    data = np.array(data)

    # Stats
    total_count = int(np.sum(data))
    n = len(data)
    sample_mean = total_count / n

    # MLE
    mle = sample_mean

    # Log-likelihoods
    log_likelihoods = {}
    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(data, lam)
        log_likelihoods[lam] = ll

    # Best candidate
    best_candidate = None
    best_value = -float("inf")

    for lam, ll in log_likelihoods.items():
        if ll > best_value:
            best_value = ll
            best_candidate = lam

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate
    }
