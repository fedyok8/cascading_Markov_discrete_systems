import enum
import typing
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import os
import sys
import json

class EventStatus(enum.Enum):
    arrive = 0
    leave = 1
    drop = 2

class Event:
    def __init__(self, time: float, status: EventStatus, queue_size: int) -> None:
        self.time       = time
        self.status     = status
        self.queue_size = queue_size
    def to_string(self) -> str:
        return str(self.time).ljust(15) + str(self.status).ljust(20) + str(self.queue_size)

class Packet:
    def __init__(self, id: int, time_arrive: float) -> None:
        self.id = id
        self.time_arrive = time_arrive
        self.time_leave = -1.0

class Server:
    def __init__(self, queue_capacity: int) -> None:
        self.memory_usage_statustic = {}
        self.next_packet_id = 0
        self.time_last_event = 0
        self.lost_packets_count = 0
        self.queue_capacity = queue_capacity
        self.processed_packets = list()
        self.queue = collections.deque()
    def push(self, time: float) -> None:
        if len(self.queue) < self.queue_capacity:
            self.update_memory_usage_stat(time)
            self.queue.append(Packet(self.next_packet_id, time))
            self.next_packet_id += 1
        else:
            self.lost_packets_count += 1
    def pop(self, time: float) -> None:
        self.update_memory_usage_stat(time)
        p = self.queue.popleft()
        p.time_leave = time
        self.processed_packets.append(p)
    def update_memory_usage_stat(self, time: float):
        duration = time - self.time_last_event
        memory_size = len(self.queue)
        if not (memory_size in self.memory_usage_statustic.keys()):
            self.memory_usage_statustic[memory_size] = 0
        self.memory_usage_statustic[memory_size] += duration
        self.time_last_event = time

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


def get_queue_capacity(evenst: typing.List[Event]) -> int:
    capasity = 0
    for e in evenst:
        if e.queue_size > capasity:
            capasity = e.queue_size
    return capasity

def get_experiment_results(datfile_path = 'mm1queue.dat') -> typing.List[Event]:
    events = list()
    with open(datfile_path) as ifstream:
        for line in ifstream:
            time_str, sign_str, count_str = line.rstrip().split()
            status = None
            if sign_str == '+':
                status = EventStatus.arrive
            elif sign_str == '-':
                status = EventStatus.leave
            elif sign_str == 'd':
                status = EventStatus.drop
            else:
                raise ValueError(f'ERROR: undefiend status string = {sign_str}')
            events.append(Event(float(time_str), status, int(count_str)))
    return events

def run_transmittion(events: typing.List[Event]) -> Server:
    queue_capacity = get_queue_capacity(events)
    server = Server(queue_capacity)
    for e in events:
        if e.status in [EventStatus.arrive, EventStatus.drop]:
            server.push(e.time)
        elif e.status == EventStatus.leave:
            server.pop(e.time)
        else:
            raise ValueError(f'ERROR: undefiend status string = {e.status}')
    for i, p in enumerate(server.processed_packets):
        if p.id != i:
            raise ValueError(f'ERROR: packet id {p.id} != {i} packet position')
    return server

def plot_bar(x: list, y: list, plt_name: str, x_label="x", y_label="y"):
    bar_width = x[1] - x[0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    ax.bar(x, y, width=bar_width*0.8)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", linestyle="-", color="black", linewidth=0.7)
    ax.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
    ax.tick_params(which='major', length=10, width=1, direction="in")
    ax.tick_params(which='minor', length=5, width=0.7, direction="in")
    fig.suptitle(plt_name, fontsize=16)
    fig.savefig(f'{plt_name}.png')
    plt.show()

def get_mean_std_from_mem_usage_stat(stat: dict) -> typing.Tuple[float, float]:
    mean = sum([mem_size * time for mem_size, time in stat.items()]) / sum(stat.values())
    mean_of_squared = sum([mem_size * mem_size * time for mem_size, time in stat.items()]) / sum(stat.values())
    std = mean_of_squared - mean * mean
    return mean, std

def main(datfile_path):
    experiment_name = os.path.split(datfile_path)[1][:-4]
    events = get_experiment_results(datfile_path)
    server = run_transmittion(events)
    durations = np.array([p.time_leave - p.time_arrive for p in server.processed_packets], dtype=np.float64)
    histogram = Histogram(durations, 100)

    plot_bar(histogram.x_centre,
        histogram.y,
        f"process_time_distribution_{experiment_name}",
        "packet_duration [s]",
        "number_of_packets"
    )
    plot_bar(sorted(list(server.memory_usage_statustic.keys())),
        [server.memory_usage_statustic[i] for i in sorted(list(server.memory_usage_statustic.keys()))],
        f"memory_size_distribution_{experiment_name}",
        "memory_size",
        "duration [s]"
    )

    result_dict = {"packet_durations": {"mean": durations.mean(), "std": durations.std()}}
    result_dict["memory_usage_statustic"] = server.memory_usage_statustic
    mean_of_mem_size, std_of_mem_size = get_mean_std_from_mem_usage_stat(server.memory_usage_statustic)
    result_dict["mem_size"] = {"mean": mean_of_mem_size, "std": std_of_mem_size}
    result_dict["lost_packets_count"] = server.lost_packets_count
    result_dict["processed_packets_count"] = len(server.processed_packets)
    with open(f"{experiment_name}.json", "w") as ofstream:
        json.dump(result_dict, ofstream, ensure_ascii=True, indent=4)

if __name__ == '__main__':
    datfile_path = "mm1queue.dat"
    datfile_path = 'dat/mm1q_mu-1.001_queueSize-100000.dat'
    if len(sys.argv) > 1:
        datfile_path = sys.argv[1]
    if not (os.path.isfile(datfile_path) and datfile_path.endswith(".dat")):
        print("skip", sys.argv)
    else:
        main(datfile_path)
        print(f"DONE: {datfile_path}")
