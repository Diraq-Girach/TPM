import numpy as np

from tools import hebbian


class TPM:
    def __init__(self, k=None, n=None, l=None, *, weights_generator=None, signum=None, optimization=hebbian):
        self.optimization = optimization
        assert k and n and l, "Specify the shape of TPM"
        assert signum, "Provide signum function"
        self.k = k
        self.n = n
        self.l = l
        self.signum = signum
        self.weights = weights_generator(k, n, l) if weights_generator else np.random.randint(-l + 1, l + 1, size=(k, n))

    def get_output(self, vec):
        vec = vec.reshape(self.k, self.n)
        tmp = np.multiply(vec, self.weights)
        tmp = np.sum(tmp, axis=1)
        sigma = self.signum(tmp)
        tau = np.prod(sigma)
        return tau, sigma

    def optimize(self, x, remote_tau):
        this_tau, sigma = self.get_output(x)

        self.weights = self.optimization(self.weights, x, sigma, this_tau, remote_tau, -self.l, self.l)

    def get_key(self):
        return self.weights.reshape(-1, 1)
