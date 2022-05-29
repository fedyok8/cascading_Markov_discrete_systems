import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.optimize import curve_fit


def derivation(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    derivation_x = x[:-1]
    derivation_y = dy / dx
    return derivation_x, derivation_y

def moving_mean_n(x, n = 2):
    len_x = x.shape[0]
    if len_x < 4*n:
        print('WARNING n too big:', n, len_x)
        return x
    arr_mean = np.zeros((len_x - n*2,), dtype=np.float64)

    # list_i = list(range(n, len_x - n))
    for i in range(n, len_x - n):
        summa = 0
        counter = 0
        # list_j = list(range(i-n, i+n+1))
        for j in range(i-n, i+n+1):
            summa += x[j]
            counter += 1
        arr_mean[i - n] = summa / counter
    return arr_mean


# arr_src = [0, 1, 2, 4, 8, 16, 32, 100, 200, 10000]
# arr_dst = list(moving_mean_n(np.array(arr_src)))
# _=0


def plot(x: np.ndarray, y: np.ndarray, plt_name: str, x_label="x", y_label="y"):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(9, 9))
    text = ''
    for i in range(1, 5):
        text += f'M{i}^(1/{i})={round(calc_moment(x, y, i), 2)}; '
    
    fig.suptitle(f'{plt_name}\n{text}', fontsize=16)
    for a in ax:
        a.set_xlabel(x_label, fontsize=10)
        a.set_ylabel(y_label, fontsize=10)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.grid(which="major", linestyle="-", color="black", linewidth=0.7)
        a.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
        a.tick_params(which='major', length=10, width=1, direction="in")
        a.tick_params(which='minor', length=5, width=0.7, direction="in")
    fourier = np.abs(np.fft.fft(y))
    derivation_x, derivation_y = derivation(x, y)
    derivation2_x, derivation2_y = derivation(derivation_x, derivation_y)
    [param_C, param_A], res1 = curve_fit(lambda x1, param_C, param_A: param_C * np.exp(-param_A * x1), x, y)
    approx = param_C * np.exp(-param_A * x)
    ax[0].plot(x, y)
    ax[0].plot(x, approx, 'r--')

    ax[0].set_ylabel('y', fontsize=10)
    ax[1].plot(derivation_x, derivation_y)
    ax[1].set_ylabel('dy/dx', fontsize=10)
    ax[2].plot(x, y - approx)
    ax[2].set_ylabel('y-yaprox', fontsize=10)    
    
    fig.savefig(f'plots/memory_size_analys/{plt_name}.png')
    # plt.show()

def calc_moment(x: np.ndarray, px: np.ndarray, moment_order) -> float:
    px_normed = px / np.sum(px)
    x_powered_to_n = np.power(x, moment_order)
    moment = np.sum(px_normed * np.power(x, moment_order))
    return np.power(moment, 1/moment_order)




def main(jsonfile_path):
    experiment_name = os.path.split(jsonfile_path)[1][:-5]
    with open(jsonfile_path) as ifstream:
        info = json.load(ifstream)
    data = [(int(mem_size), time) for mem_size, time in info['memory_usage_statustic'].items()]
    data_sorted = sorted(data, key = lambda x: x[0])
    x = np.array([d[0] for d in data_sorted], dtype=np.int32)
    y = np.array([d[1] for d in data_sorted], dtype=np.float64)
    y_normed = y / np.sum(y)
    
    plot(x, y_normed, f'{experiment_name}', 'mem_size', 'duration')
    # fourier = np.abs(np.fft.fft(derivation_y))
    # plot(derivation_x, fourier, f'fourier_{experiment_name}', 'freq', 'fourier')
    return

if __name__ == '__main__':
    jsonfile_path = 'json/mm1q_mu-1.001_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.01_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.05_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.1_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.2_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.3_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.5_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-1.7_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-2_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-2.5_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-3_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-5_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-7_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-4_queueSize-100000.json'
    # jsonfile_path = 'json/mm1q_mu-10_queueSize-100000.json'

    if len(sys.argv) > 1:
        jsonfile_path = sys.argv[1]
    if not (os.path.isfile(jsonfile_path) and jsonfile_path.endswith(".json")):
        print("skip", sys.argv)
    else:
        main(jsonfile_path)
        print(f"DONE: {jsonfile_path}")
