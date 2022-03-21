import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from activations import *

class MLP:
    #layers_dim = numero de neuronas en cada capa,numero de entradas y numero de salidas, en una tupla
    def __init__(self, layers_dim, activations):
        
        # Atributos
        self.W = [None]
        self.b = [None]
        self.f = [None]
        self.n = layers_dim
        self.L = len(layers_dim)-1

        for l in range(self.L):
            self.W.append(-1 +2 * np.random.rand(self.n[l+1], self.n[l]))
            self.b.append(-1 +2 * np.random.rand(self.n[l+1], 1))
        
        for act in activations:
            self.f.append(activate(act))

    def backpropagation(self, x):
        a = np.asanyarray(x)
        for l in range (1, self.L + 1):
            z = np.dot(self.W[l], a) + self.b[l]
            a = self.f[l](z)
        return a
    def train(self, x,y, epochs=1000, learning_rate=0.1):
        x = np.asanyarray(x)
        y = np.asanyarray(y).reshape(self.n[-1], 1)

        P = x.shape[1]

        for _ in range(epochs):
            for p in range(P):

                A = [None] * (self.L + 1)
                dA = [None] * (self.L + 1)

                #Propagacion
                A[0] = x[:, p].reshape(self.n[0], 1)

                for l in range(1, self.L +1):
                    z = np.dot(self.W[l], A[l-1]) + self.b[l]
                    A[l], dA[l] = self.f[l](z, derivative=True)
                
                #Backpropagation
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        lg = (y[:, p] - A[l]) * dA[l]
                    else:
                        lg = np.dot(self.W[l+1].T, lg) * dA[l]



