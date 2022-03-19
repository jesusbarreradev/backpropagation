import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def sigmoid(z):
    return (1 / (1 + np.exp(-1 * z)))

class MLP():
    def __init__(self, i_neurons, h_layers, h_neurons, o_neurons):
        # número de neuronas por capa
        self.input_neurons = i_neurons
        self.hidden_neurons = h_neurons
        self.output_neurons = o_neurons

        # número de capas ocultas
        self.hidden_layers = h_layers - 1

        # tasa de aprendizaje
        self.lr = 0.1

        # Función para imprimir el error cuadrático medio
        self.error_figure = None

        # guarda los valores de activación de todas las capas
        self.sigmoids = list(range(2 + self.hidden_layers))

        # guarda las sensibilidades = input layer + output layer + hidden layers
        self.sensitivities = list(range(h_layers + 1))

        # matrices de pesos
        # pesos de la entrada a la primer capa oculta
        self.W_inputs = np.empty((self.hidden_neurons, self.input_neurons + 1))
        # pesos de las capas ocultas menos la última
        self.W_hiddens = np.empty((self.hidden_layers, self.hidden_neurons, self.hidden_neurons + 1))
        # pesos de la última capa oculta y la capa final
        self.W_outputs = np.empty((self.output_neurons, self.hidden_neurons + 1))
        self.randomize_weights()
        
 def backpropagation():
        """Realiza el método de retropropagación"""
        # actualiza los pesos de la capa oculta final con la capa de salida