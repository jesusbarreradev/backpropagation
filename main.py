import numpy as np 
from matplotlib import pyplot as plt 

def mouse_event(event): 
    print('x: {} and y: {}'.format(event.xdata, event.ydata))


fig = plt.figure() 
cid = fig.canvas.mpl_connect('button_press_event', mouse_event) 
#x = np.linspace(-5, 5, 10) 
#y = np.sin(x) 
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.axhline(0, color='k', lw=1)
plt.axvline(0, color='k', lw=1)
#plt.plot(x, y) 
plt.show()
