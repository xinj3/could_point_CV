from webbrowser import get
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display
import time
from matplotlib.widgets import Slider, TextBox
import os
data_frames = []

with open("range_noise_fft_old.log") as f:
    data_frames = f.readlines()

# plt.ion()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
x = np.arange(256)
y = np.ones(256) * 15
y[255] = 45
l, = plt.plot(x, y)
l.set_label('range')

m, = plt.plot(x, y)
m.set_label('noise')
ax.legend()


all_noise = []
all_range = []
timestamps = []
len_data = len(data_frames)


for frame in data_frames:
    ts_idx = frame.find('"ts": ')
    comma_idx = frame.find(',')
    ts = frame[ts_idx + 6:comma_idx]
    timestamps.append(ts)
    range_idx = frame.find("range_profile")
    range_and_noise = frame[range_idx:]
    range = range_and_noise[:range_and_noise.find("]")]
    range_arr = range[range.find("[") + 1:].split(", ")
    noise = range_and_noise[range_and_noise.find("]"):]
    noise = noise[noise.find("[") + 1:]
    noise_arr = noise[:noise.find("]")].split(", ")
    all_noise.append(noise_arr)
    all_range.append(range_arr)


ax_ts = plt.axes([0.25, 0.15, 0.40, 0.03])
ax_box = plt.axes([0.25, 0.10, 0.40, 0.03])
ax_message = plt.axes([0.25, 0.05, 0.60, 0.03])
text_box = TextBox(ax_box, "enter timestamp")
message_box = TextBox(ax_message, "message")

ts_num = Slider(ax_ts, 'timestamp slider', float(timestamps[0]),
                float(timestamps[len(timestamps) - 1]), float(timestamps[0]))


def closest(lst, K):
    if K == '':
        return -2
    try:
        K = float(K)
    except:
        return -1
    lst = np.array(lst).astype('float64')
    if K < lst[0] or K > lst[len(lst) - 1]:
        return -1
    idx = (np.abs(lst - K)).argmin()
    return idx


def get_plot(idx):

    pd_data = pd.DataFrame(
        {'noise': all_noise[idx], 'range': all_range[idx]})
    pd_data['noise'] = pd_data['noise'].astype(float)
    pd_data['range'] = pd_data['range'].astype(float)

    y_range = pd_data['range']
    y_noise = pd_data['noise']
    # plt.title(ts)
    l.set_ydata(y_range)
    m.set_ydata(y_noise)
    # ax.cla()


def get_num_slider(val):
    idx = closest(timestamps, float(ts_num.val))
    message_box.set_val("displayed timestamp: " + timestamps[idx])
    text_box.set_val(timestamps[idx])
    get_plot(idx)


def get_num_box(val):
    idx = closest(timestamps, val)
    if idx == -1:
        message_box.set_val("invalid timestamp entered")
    elif idx == -2:
        return
    else:
        message_box.set_val("displayed timestamp: " + timestamps[idx])
        get_plot(idx)


text_box.on_submit(get_num_box)

ts_num.on_changed(get_num_slider)
plt.show()

