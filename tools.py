import numpy as np


class SetterProperty(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)

@np.vectorize
def signum_zero_to_plus(a):
    """
    Return signum without zero
    :param a: number
    :return: signum valueint
    """
    return 1 if a >= 0 else -1


@np.vectorize
def signum_zero_to_minus(a):
    """
    Return signum without zero
    :param a: number
    :return: signum valueint
    """
    return 1 if a >= 1 else -1


@np.vectorize
def theta(a, b):
    return int(a == b)


@np.vectorize
def negative_theta(a, b):
    return int(a != b)


def hebbian(weights, x, sigma, tau, remote_tau, min_l, max_l):
    x = x.reshape(weights.shape)
    sigma = np.tile(sigma.reshape(-1, 1), (1, weights.shape[1]))  # repeats array
    if remote_tau != tau:
        return weights
    return np.clip(
        weights + x * tau * theta(sigma, tau) * theta(tau, remote_tau), min_l, max_l
    )


def anti_hebbian(weights, x, sigma, tau, remote_tau, min_l, max_l):
    x = x.reshape(weights.shape)
    sigma = np.tile(sigma.reshape(-1, 1), (1, weights.shape[1]))  # repeats array
    #
    # rate = sigma * x * theta(sigma, tau_a) * theta(tau_a, tau_b)
    # print(rate)
    # print("RATE: ")

    return np.clip(
        weights - x * tau * theta(sigma, tau) * theta(tau, remote_tau), min_l, max_l
    )


