import numpy as np 
from matplotlib import pyplot as plt 
#plt.rcParams["figure.figsize"] = [7.00, 3.50] 

def mouse_event(event): 
    print('x: {} and y: {}'.format(event.xdata, event.ydata))


fig = plt.figure() 
cid = fig.canvas.mpl_connect('button_press_event', mouse_event) 
x = np.linspace(-5, 5, 10) 
y = np.sin(x) 
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.plot(x, y) 
plt.show()
