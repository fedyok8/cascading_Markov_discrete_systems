from cmath import nan
import sys
import os
import json
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.optimize import curve_fit
import math

class Histogram:
    def __init__(self, events: np.ndarray, subdivisions_count = 100, jitter = 0.00001) -> None:
        self.subdivisions_count = subdivisions_count
        self.jitter = jitter
        self.val_max = events.max()
        self.val_min = events.min()
        self.step_size = (self.val_max - self.val_min) * (1 + jitter) / subdivisions_count
        anchor_points = np.array([self.val_min + i * self.step_size for i in range(subdivisions_count + 1)], dtype=float)
        self.x_left = anchor_points[:-1]
        self.x_right = anchor_points[1:]
        self.x_centre = (self.x_left + self.x_right) / 2
        self.y = np.zeros((subdivisions_count), dtype=np.int32)
        for i in range(subdivisions_count):
            self.y[i] = sum([1 for d in events if d < self.x_right[i] and d >= self.x_left[i]])
        self.integral = np.sum(self.y)
        self.y_normed = np.array(self.y/self.integral, dtype=float)
        self.integral_normed = np.sum(self.y_normed)
        self.mean = np.sum(self.x_centre * self.y_normed)
        x_centre_squared = self.x_centre **2
        mean_of_squared = np.sum(x_centre_squared * self.y_normed)
        self.variance = mean_of_squared - self.mean **2
        self.standard_deviation = np.sqrt(self.variance)

class Moments:
    def __init__(self, x: np.ndarray, px: np.ndarray) -> None:
        if len(x.shape) > 1:
            print("ERROR: shape must be (N,) but is:", x.shape)
            return
        if len(px.shape) > 1:
            print("ERROR: shape must be (N,) but is:", x.shape)
            return
        if not (x.shape == px.shape):
            print("ERROR: different shapes", x.shape, px.shape)
            return
        self.raw = {
            moment_order:
            calc_moment(x, px, moment_order) for moment_order in range(1, 5)
        }
        self.central = {
            moment_order:
            calc_moment(x - self.raw[1], px, moment_order) for moment_order in range(1, 5)
        }
        for k, v in self.central.items():
            if np.isnan(v):
                _=0

def my_exp(x1, C0, alpha):
    return C0 * np.exp(-1 * alpha * x1)

class ExponentApprox:
    def __init__(self, x, y) -> None:
        [param_C, param_A], self.covariance = curve_fit(my_exp, x, y)
        self.c0 = param_C
        self.alpha = param_A
    def calc_value(self, x: float) -> float:
        return self.c0 * math.exp(-1 * self.alpha * x)

def calc_moment(x: np.ndarray, px: np.ndarray, moment_order) -> float:
    px_normed = px / np.sum(px)
    moment = np.sum(px_normed * np.power(x, moment_order))
    if moment < 0 and (moment_order % 2 == 1):
        moment_same_dim = -np.power(-moment, 1/moment_order)
    else:
        moment_same_dim = np.power(moment, 1/moment_order)
    return moment_same_dim

def get_mu_and_qSize_from_filename(filepath):
    filename = os.path.split(filepath)[1]
    if not (filename.startswith("mm1q_mu-") and filename.endswith(".json")):
        return 0
    filename_withot_ext = filename[:-5]
    _, str_mu, str_qsize = filename_withot_ext.split('_')
    mu = float(str_mu.split('-')[1])
    qsize = int(str_qsize.split('-')[1])
    return mu, qsize

class ExperimentResult:
    def __init__(self, json_file_path) -> None:
        with open(json_file_path) as ifstream:
            info = json.load(ifstream)
        data = [(int(mem_size), time) for mem_size, time in info['memory_usage_statustic'].items()]
        data_sorted = sorted(data, key = lambda x: x[0])
        self.arr_mem_size = np.array([d[0] for d in data_sorted], dtype=np.int32)
        self.arr_duration = np.array([d[1] for d in data_sorted], dtype=np.float64)
        self.arr_prob = self.arr_duration / np.sum(self.arr_duration)
        self.mu, self.queue_size = get_mu_and_qSize_from_filename(json_file_path)
        self.lambda0 = 1
        self.moments = Moments(self.arr_mem_size, self.arr_prob)
        self.approximation = ExponentApprox(self.arr_mem_size, self.arr_prob)
        approx_function_vectorized = np.vectorize(self.approximation.calc_value)
        self.arr_prob_theoretical = approx_function_vectorized(self.arr_mem_size)
        self.arr_deviation = np.abs(self.arr_prob - self.arr_prob_theoretical)/(self.arr_prob_theoretical.max())
        self.deviation_hist = Histogram(self.arr_deviation)
        self.moments_deviation = Moments(self.deviation_hist.x_centre, self.deviation_hist.y_normed)
        _=0


def plot(results: typing.List[ExperimentResult]):
    results = sorted(results, key = lambda x: x.mu)
    x = [r.mu/r.lambda0-1 for r in results]
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
    fig.suptitle("mm1c moments", fontsize=16)    
    for a in ax:
        a.set_xlabel('mu/lambda', fontsize=10)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.grid(which="major", linestyle="-", color="black", linewidth=0.7)
        a.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
        a.tick_params(which='major', length=10, width=1, direction="in")
        a.tick_params(which='minor', length=5, width=0.7, direction="in")
    for mom_ord in range(1, 5):
        y_raw = [r.moments.raw[mom_ord] for r in results]
        y_centr = [r.moments.central[mom_ord] for r in results]
        ax[0].plot(x, y_raw, "o-", label=f"M{mom_ord}")
        if mom_ord > 1:
            ax[1].plot(x, y_centr, "o-", label=f"M{mom_ord}")
    ax[0].set_ylabel("raw_moment", fontsize=10)
    ax[1].set_ylabel("central_moment", fontsize=10)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    fig.savefig("mm1c_moments.png")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
    fig.suptitle("mm1c deviation moments", fontsize=16)
    for a in ax:
        a.set_xlabel('mu/lambda', fontsize=10)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.grid(which="major", linestyle="-", color="black", linewidth=0.7)
        a.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
        a.tick_params(which='major', length=10, width=1, direction="in")
        a.tick_params(which='minor', length=5, width=0.7, direction="in")
    for mom_ord in range(1, 5):
        y_raw = [r.moments_deviation.raw[mom_ord] for r in results]
        y_centr = [r.moments_deviation.central[mom_ord] for r in results]
        ax[0].plot(x, y_raw, "o-", label=f"M{mom_ord}")
        if mom_ord > 1:
            ax[1].plot(x, y_centr, "o-", label=f"M{mom_ord}")
    ax[0].set_ylabel("raw_moment", fontsize=10)
    ax[1].set_ylabel("central_moment", fontsize=10)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    fig.savefig("mm1c_deviation_moments.png")
    plt.show()
    plt.close(fig)
    return





def main(dir1 = "json"):
    if not os.path.isdir(dir1):
        print("ERROR: is not dir", dir1)
        return
    results = list()
    for jfile in os.listdir(dir1):
        mu, que_size = get_mu_and_qSize_from_filename(jfile)
        if que_size < 100000:
            continue
        elif que_size == 100000:
            result = ExperimentResult(os.path.join(dir1, jfile))
            results.append(result)
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
            
            # ax.plot(result.arr_mem_size, result.arr_prob)
            # ax.plot(result.arr_mem_size, result.arr_prob_theoretical, "--")
            # fig.show()
            # plt.close(fig)
        else:
            print("ERROR: quequ size shoul be <= 10^5", jfile)
            return
    plot(results)
    _=0


if __name__ == '__main__':
    main("json/")
