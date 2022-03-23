from backpropagation import *
import matplotlib.pyplot as plt

x = [[0,0,1,1],
    [0,1,0,1]]

y = [1,0,0,1]

net = MLP((2,2,1), ('tanh', 'sigmoid'))
#salida antes de entrenar
print(net.backpropagation(x))

net.train(x,y, epochs=10000, learning_rate=0.5)
#salida con el entrenamiento, esta es mas parecida a y = [1,0,0,1]
print(net.backpropagation(x))


plt.figure()
#para testearlo dibujo puntos a mano
plt.plot(0,0,'b*')
plt.plot(1,1,'b*')
plt.plot(0,1,'r*')
plt.plot(1,0,'r*')

xx, yy = np.meshgrid(np.arange(-5,5.1,0.1),np.arange(-5,5.1,0.1))
x_input = [xx.ravel(),yy.ravel()]
zz = net.backpropagation(x_input)
zz = zz.reshape(xx.shape)

plt.contourf(xx,yy,zz, alpha=0.8, cmap=plt.cm.RdBu)

plt.title("MLP entrenada con Backpropagation")
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.grid()
plt.show()