"""
Necesito aprender a utilizar los arrays de numpy
para agregar las coordenadas de cada nuevo clic.
Actualizar el canvas para pintar cada uno de los
puntos.
"""


from tkinter import *
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
from matplotlib import pyplot as plt
from helpers import *
from Point import *

data = []
point = Point(0,0)
"""
data.append(point.x)
data.append(point.y)
"""

def update_canvas():
    _data = np.array(data)
    x, y = _data.T
    plt.scatter(x,y)


def mouse_event(event): 
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    point.x = round(event.xdata, 3)
    point.y = round(event.ydata, 3)
    data.append(point)
    update_canvas()
    #np.insert(data, round(event.xdata, 3), round(event.ydata, 3))


fields = 'Learning rate', 'Error minimo', 'Epocas maximas'
buttons = 'Iniciar Backpropagation', 'Iniciar pesos'
padding = 5
fig, ax = plt.subplots(facecolor='#fff')
cid = fig.canvas.mpl_connect('button_press_event', mouse_event) 
#vela = fig.canvas.mpl_connect(update_canvas) 

plt.ylim([-5,5])
plt.xlim([-5,5])
plt.axhline(0, color='k', lw=1)
plt.axvline(0, color='k', lw=1)

if __name__ == '__main__':
    window = Tk()
    window.geometry('700x600')
    window.wm_title('Backpropagation')

    frame = Frame(window, bg='white')
    frame.pack(expand=1, fill='both')

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(padx=padding, pady=padding, expand=1 ) #fill='both'

    _formfields = makeformfields(window, fields)
    _capas_spinbox = makespinbox(window, "Capas ocultas", 1, 2)
    _neuronas_spinbox = makespinbox(window, "Neuronas/p", 1, 10)
    _botones = makebuttons(window, buttons)
    window.mainloop()

