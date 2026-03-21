import numpy as np
import matplotlib.pyplot as plt

def get_volume(distance):
    volume = int(np.interp(distance, [20,200], [0,100]))
    return max(0, min(100, volume))

def draw_mapping_graph(distance, volume):
    fig, ax = plt.subplots(figsize=(4,2))

    x_line = np.linspace(20,200,50)
    y_line = np.interp(x_line,[20,200],[0,100])

    ax.plot(x_line, y_line)
    ax.scatter(distance, volume, s=120)

    ax.set_xlim(0,200)
    ax.set_ylim(0,100)
    ax.set_title("Distance → Volume Mapping")

    return fig

def draw_history(history):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.plot(history)
    ax.set_title("Volume History")
    return fig