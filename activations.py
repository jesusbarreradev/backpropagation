#Funciones de activacion 
import numpy as np

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 + a) * (1 - a)
        return a, da
    return a

def sigmoid(z, derivative=False):
    a = 1 / (1 + np.exp(-z)) 
    if derivative:
        da = a * (1 - a)
        return a, da
    return a