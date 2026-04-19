import math
import numpy as np
import matplotlib.pyplot as plt
import re


def plot_acc_w_segment(data_dict, img_pth):
    for key, value in data_dict.items():
        if math.isnan(sum(value)) or len(value) == 0:  # Check for NaN or empty list
            data_dict[key] = 0
        else:
            data_dict[key] = sum(value) / len(value)
        if data_dict[key] < 0:
            data_dict[key] = 0
    sorted_keys = sorted(data_dict.keys())
    x = sorted_keys
    y = [data_dict[k] for k in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-') 
    
    for x_val, y_val in zip(x, y):
        plt.text(x_val, y_val, f"{y_val}", 
                 ha='center', va='bottom', 
                 fontsize=9, color='black')  

    plt.xlabel('Number of Segments')
    plt.ylabel('Predict Acc %')
    plt.grid(True)
    plt.savefig(img_pth, dpi=300, bbox_inches='tight')
    plt.show()


def process_glitch_wave(measurements):
    data = {}
 
    def parse_voltage_wave(columns, lines):
        data = {}
        
        for line in lines:
            values = [float(value) for value in line.split()[1:]]
            for idx, column in enumerate(columns):
                if column not in list(data.keys()):
                    data[column] = []
                else:
                    data[column].append(values[idx])
        return data

    with open(measurements, "r") as file:
        lines = file.readlines()
    with open(measurements, "r") as file:
        for index, line in enumerate(file, start=1):
            if "time" in line and "voltage" in line:
                column_start = index
                data_start= column_start + 1
                data_end = column_start + 42
                chunk = lines[data_start:data_end]
                chunk_parse = parse_voltage_wave(lines[column_start].split(), chunk)
                data.update(chunk_parse)

    return data


def extract_glitch_metric(wave):
    voltage = np.array(wave)
    dt = 5.0e-11
    time = np.arange(0, len(voltage)) * dt
    v_max_p = np.max(voltage)
    v_max_n = np.min(voltage)
    half_thresh_p = 0.5 * v_max_p
    half_thresh_n = 0.5 * v_max_n
    def find_crossings(t, v, thr):
        """
        Find times where v crosses 'thr' (by linear interpolation).
        Returns an array of crossing times (in seconds).
        
        We look for consecutive indices i, i+1 where v[i] < thr and v[i+1] > thr (rising),
        or v[i] > thr and v[i+1] < thr (falling). Then we solve linearly for the
        crossing time between t[i] and t[i+1].
        """
        crossings = []
        for i in range(len(v) - 1):
            v1, v2 = v[i], v[i+1]
            t1, t2 = t[i], t[i+1]
            # (one side above thr, the other side below thr)
            if (v1 - thr) * (v2 - thr) < 0:
                # Linear interpolation:
                # v1 + alpha*(v2 - v1) = thr  =>  alpha = (thr - v1)/(v2 - v1)
                alpha = (thr - v1) / (v2 - v1)
                crossing_time = t1 + alpha*(t2 - t1)
                crossings.append(crossing_time)
        return np.array(crossings)
    positive_crossings = find_crossings(time, voltage, half_thresh_p) # positive glitch
    negative_crossings = find_crossings(time, voltage, half_thresh_n) # positive glitch
    tw_p = (positive_crossings[1] - positive_crossings[0])*1e09
    tw_n = (negative_crossings[1] - negative_crossings[0])*1e09
    return [(v_max_p, tw_p), (v_max_n, tw_n)]


