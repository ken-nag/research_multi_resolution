import matplotlib.pyplot as plt
import numpy as np

def plot_spec(spec, fs, sec, vmax, vmin):
    t = get_time(spec, sec)
    f = get_freq(spec, fs)
    fig = plt.figure()
    ax  =fig.add_subplot(1,1,1)
    ax_spec = ax.pcolormesh(t, f, 20*np.log10(np.abs(spec.T) + 1e-14), vmin=vmin, vmax=vmax)
    pp = fig.colorbar(ax_spec, ax=ax)
    pp.set_clim(vmin,  vmax)
    plt.show()


def plot_log_spec(spec, fs, sec, vmax, vmin):
    t = get_time(spec, sec)
    f = get_freq(spec, fs)
    fig = plt.figure()
    ax  =fig.add_subplot(1,1,1)
    ax_spec = ax.pcolormesh(t, f, spec.T , vmin=vmin, vmax=vmax)
    pp = fig.colorbar(ax_spec, ax=ax)
    pp.set_clim(vmin,  vmax)
    plt.show()



def get_time(spec, sec):
    return np.arange(0, sec, sec/spec.shape[0])  

def get_freq(spec, fs):
    return np.arange(0, fs/2, (fs/2)/spec.shape[1])
