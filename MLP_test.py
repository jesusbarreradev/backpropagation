from backpropagation import *


x = [[0,0,1,1],
    [0,1,0,1]]

y = [1,0,0,1]

net = MLP((2,50,1), ('tanh', 'sigmoid'))
#salida antes de entrenar
print(net.backpropagation(x))

net.train(x,y, epochs=10000, learning_rate=0.5)
#salida con el entrenamiento, esta es mas parecida a y = [1,0,0,1]
print(net.backpropagation(x))