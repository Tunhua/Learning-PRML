import matplotlib.pyplot as plt

import numpy as np

from scipy.special import gamma, digamma


class Gaussian(object):

    def fit(self, x):

        self.mean = np.mean(x)

        self.var = np.var(x)

    def predict_probability(self, x):

        return (np.exp(-0.5 * (x - self.mean) ** 2 / self.var)

                / np.sqrt(2 * np.pi * self.var))


class StudentT(object):

    def __init__(self, mean=0, a=1, b=1, learning_rate=0.01):

        self.mean = mean

        self.a = a

        self.b = b

        self.learning_rate = learning_rate

    def fit(self, x):

        while True:

            params = [self.mean, self.a, self.b]

            self._expectation(x)

            self._maximization(x)

            if np.allclose(params, [self.mean, self.a, self.b]):

                break

    def _expectation(self, x):

        self.precisions = (self.a + 0.5) / (self.b + 0.5 * (x - self.mean) ** 2)

    def _maximization(self, x):

        self.mean = np.sum(self.precisions * x) / np.sum(self.precisions)

        a = self.a

        b = self.b

        self.a = a + self.learning_rate * ( len(x) * np.log(b) + np.log(np.prod(self.precisions)) - len(x) * digamma(a))

        self.b = a * len(x) / np.sum(self.precisions)

    def predict_probability(self, x):

        return ((1 + (x - self.mean) ** 2/(2 * self.b)) ** (-self.a - 0.5) * gamma(self.a + 0.5) / (gamma(self.a) * np.sqrt(2 * np.pi * self.b)))


def main():

    x = np.random.normal(size=20)

    x = np.concatenate([x, np.random.normal(loc=20, size=3)])

    plt.hist(x, bins=50, normed=1, label="samples")


    students_t = StudentT()
    gaussian = Gaussian()


    students_t.fit(x)
    gaussian.fit(x)


    x = np.linspace(-5, 25, 1000)
    plt.plot(x, students_t.predict_probability(x), label="student's t", linewidth=2)
    plt.plot(x, gaussian.predict_probability(x), label="gaussian", linewidth=2)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    main()
