import math
from copy import copy
import abc


class Basis(abc.ABC):

    @abc.abstractmethod
    def convert_features(self, features: []):
        pass

    @abc.abstractmethod
    def dimension(self):
        pass


class LinearBasis(Basis):

    def __init__(self, dimension: int):
        """

        :param dimension: int
            dimension of the feature vector that the Fourier basis applied to
        """
        self.__dimension = dimension

    def convert_features(self, features: []):
        return features

    def dimension(self):
        return self.__dimension


class FourierBasis(Basis):
    """
    Implementation of the Fourier basis as explained in:
    G.D. Konidaris, S. Osentoski and P.S. Thomas. Value Function Approximation in Reinforcement Learning using the Fourier Basis. In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.
    This Fourier implementation uses a feature scaling, which means that all features are normalized in [0, 1] before
    applying the Fourier transformation.
    Thus, we can assume the function to be odd in [-1, 1] and only compute the cosines (b_k) as all sines (a_k} will be 0.
    """

    def __init__(self, dimension: int, order: int, allowed_coupling: int = 1):
        """

        :param dimension: int
            dimension of the feature vector that the Fourier basis is applied to
        :param order: int
            defines the order of the Fourier basis (the higher the order, the higher the value function approximation resolution)
        :param allowed_coupling: int
            maximum number of coefficients in one vector being greater than zero
            (set to 1 if there are no dependencies between the features)
        """
        self.__dimension = dimension
        self.__order = order
        self.__allowed_coupling = allowed_coupling
        self.__coefficients = []
        self.__generate_coefficients__(0, [0] * self.__dimension, 0)

    def __generate_coefficients__(self, coefficient_index: int, vector: [], coefficients_greater_zero: int):
        # base case is we're at the end of the vector
        if coefficient_index == self.__dimension:
            self.__coefficients.append(copy(vector))
            return
        # otherwise, consider all possible values for this vector providing we don't have too many non-zero entries
        if coefficients_greater_zero >= self.__allowed_coupling:
            vector[coefficient_index] = 0
            self.__generate_coefficients__(coefficient_index+1, vector, coefficients_greater_zero)
        else:
            # consider all possible values
            for val in range(self.__order + 1):
                vector[coefficient_index] = val
                if val > 0:
                    self.__generate_coefficients__(coefficient_index+1, vector, coefficients_greater_zero+1)
                else:
                    self.__generate_coefficients__(coefficient_index+1, vector, coefficients_greater_zero)

    def convert_features(self, features: []):
        """
        Returns the basis function value for the given feature vector
        :param features: []
        :return:
        """
        fourier_basis = []
        for index in range(len(self.__coefficients)):
            current_coefficients = self.__coefficients[index % len(self.__coefficients)]
            # cross product
            sum = 0.0
            for i in range(len(features)):
                try:
                    sum += features[i] * current_coefficients[i]
                except Exception as e:
                    print(e)
            # add converted feature to fourier basis
            fourier_basis.append(round(math.cos(sum * math.pi), 2))
        # output final basis
        return fourier_basis

    def dimension(self):
        """
        The dimension of the coefficient vector depends both on the number of features and on the coupling grade.
        The general formula for computing the dimension for a feature vector of dimension m, the Fourier order n and
        coupling grade q <= m is:
        sum_{i=0}^q binomial(m q) n^i
        minimum: 1
        maximum: (n+1)^m
        :return: int
        """
        return len(self.__coefficients)

# features = [0.5, 0.25,0.3]
#
# basis = FourierBasis(len(features), 1, 1)
# basis_c = basis.convert_features(features)
# print(basis_c)
