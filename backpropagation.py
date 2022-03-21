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

def predict(self, x):
    a = np.asanyarray(x)
    for l in range (1, self.L +1):
        z = np.dot(self.W[l], a) + self.b[l]
        a = self.f[l](z)
    return a



