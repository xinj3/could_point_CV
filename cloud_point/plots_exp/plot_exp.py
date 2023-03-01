from webbrowser import get
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display
import time
from matplotlib.widgets import Slider, TextBox
import os


x = np.linspace(-10, 10, 10000)
y = np.cos(2*np.pi*x*0.1)
z = np.sin(2*np.pi*x*0.1)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x, y, z)


ax_ts = plt.axes([0.25, 0.15, 0.40, 0.03])

ts_num = Slider(ax_ts, 'freq', 0.1, 1, 0.01)

def get_num_slider(val):
    ax.clear()
    y = np.cos(2*np.pi*x*val)
    z = np.sin(2*np.pi*x*val)
    ax.scatter(x, y, z)

ts_num.on_changed(get_num_slider)
plt.show()