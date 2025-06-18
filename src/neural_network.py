import numpy as np
from numpy.random import randn
from utils import *

# 3 layer neural network
class NeuralNetwork:
    def __init__(self, W1: np.ndarray, W2: np.ndarray,
            B1: np.ndarray, B2: np.ndarray, dims: dict[str:int]) -> None:

        self.dims = dims
        self._W1 = W1
        self._W2 = W2
        self._B1 = B1
        self._B2 = B2.ravel()

    # activation function
    def sigmoid(self, y: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(y)))

    # hidden layer
    def hidden(self, input: np.ndarray) -> np.ndarray:
        h = self.sigmoid(np.dot(self._W1,input)+self._B1.ravel())
        return h

    def output(self, input: np.ndarray) -> np.ndarray:
        #forward
        # print(input.shape)
        h = self.hidden(input) # calculate the hidden layer
        y_pred = self.sigmoid(np.dot(self._W2,h) + self._B2) # output layer
        return y_pred.ravel()
