from backpropagation import *

x = [[0,0,1,1],
    [0,1,0,1]]

y = [1,0,0,1]

net = MLP((2,50,1), ('tanh', 'sigmoid'))
print(net.predict(x))