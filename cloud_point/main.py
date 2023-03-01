from point_cloud import Point_Cloud
from webbrowser import get
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display
import time
from matplotlib.widgets import Slider, TextBox
import os

# file_name = "xin_standing_still_1min_1.93m.txt"
# file_name = "xin_and_daniel_2_person_standing_still_1.93m.txt"
file_name = "xin_moving_back_and_forth_1m-1.93m.txt"
pc_2 = Point_Cloud(path=file_name, frame_size=60, eps=0.4, min_samples=5, max=4, min=0.3)
# pc_2.plot_points(frame_id=400, cluster_id=0, remove_ids=[-1])


pc_2.plot_interactive()

