#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
from itertools import islice
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def extract_data_points(fname):
    data = dict()
    with open(fname, 'rt') as ftext:
        lines = ftext.readlines()
        
        points_num = Counter()
        
        i = None
        j = None
        s = None
        recorded = 0
        
        distances = []
        forces = []
        
        for line in lines:
            if line.startswith('#'):
                if line.startswith('# iIndex:'):
                    i_val = line.split(':')[1].strip()
                    i = int(i_val)
                elif line.startswith('# jIndex:'):
                    j_val = line.split(':')[1].strip()
                    j = int(j_val)
                    temp_indices = (i, j)
                    if temp_indices in points_num:
                        s = 1
                    else:
                        s = 0

                elif line.startswith('# recorded-num-points:'):
                    recordedval = line.split(':')[1].strip()
                    recorded = int(recordedval)
                else:
                    continue
            else:
                num = line.split()
                if len(num) == 7:
                    distances.append(float(num[0]))
                    forces.append(float(num[1]))
                if len(distances) == recorded and len(forces) == recorded:
                    d = np.array(distances, dtype=float)
                    f = np.array(forces, dtype=float)
                    data[(s, i, j)] = (d, f, recorded)
                    distances.clear()
                    forces.clear()
                    points_num[temp_indices] += 1
    return data

def longest_descending(data, tolerance=3):
    d, f = data
    longest_segment = (None, None)
    max_length = 0

    start = 0
    while start < len(d):
        segment_length = 0
        fluctuation = 0
        for current in range(start, len(d) - 1):
            if f[current] > f[current + 1]:
                fluctuation = 0
            else:
                fluctuation += 1
            
            segment_length += 1
            
            if fluctuation >= tolerance:
                break

        if segment_length > max_length:
            max_length = segment_length
            longest_segment = (d[start:start+segment_length], f[start:start+segment_length])

        start += max(segment_length, 1)

    return longest_segment

def estimate_slope(data):
    d, f, recorded = data
    
    decreasing = True
    for i in range(len(d) - 1):
        if d[i] <= d[i + 1]:
            decreasing = False
            break
    
    if decreasing:
        d = list(d)
        f = list(f)
        d.reverse()
        f.reverse()
    
    d = np.array(d)
    f = np.array(f)
    
    selected_d, selected_f = longest_descending((d, f))
    
    num = len(selected_d)
    
    sigmax = np.sum(selected_d)
    sigmay = np.sum(selected_f)
    sigma_xy = np.sum(selected_d * selected_f)
    sigmax_sqr = np.sum(selected_d * selected_d)
    
    slope = ((num * sigma_xy) - (sigmax * sigmay)) / ((num * sigmax_sqr) - (sigmax ** 2))
    baseline = (sigmay - (slope * sigmax)) / num
    
    expected_f = (slope * selected_d) + baseline
    error = selected_f - expected_f
    abs_error = np.abs(error)
    
    min1, min2 = 0, 1
    for i in range(2, len(abs_error)):
        if abs_error[i] < abs_error[min1]:
            min2 = min1
            min1 = i
        elif abs_error[i] < abs_error[min2]:
            min2 = i
    
    selected_points_d = selected_d[[min1, min2]]
    selected_points_f = selected_f[[min1, min2]]
    selected_points = (selected_points_d, selected_points_f)
    
    return slope, baseline, selected_d, selected_f, selected_points, recorded

def raw_plot_with_fit(point, curve, slope, baseline, selected_d, selected_f, selected_points, recorded, save=None, show=True):
    """plot one raw distance-force curve"""
    s, i, j = point
    d, f = curve[:2]
    
    selected_points_d = selected_points[0]
    selected_points_f = selected_points[1]
    
    plt.figure(figsize=[9, 6])
    plt.plot(d, f, 'g', lw=1.8, label=f'Data: push at ({i}, {j})')
    plt.plot(selected_d, (slope * selected_d + baseline) , 'r--', lw=1.5, label=f'Slope: {slope:.5f} N/m')
    plt.scatter(selected_points_d, selected_points_f, color='red', marker='x')
    
    plt.title(f'Push at ({i}, {j}); number of records: {recorded}', fontdict={'fontsize': 14, 'fontweight': 'bold', 'family': 'sans-serif'})
    plt.xlabel("Distance (m)", fontdict={'fontsize': 10, 'fontweight': 'bold', 'family': 'sans-serif'})
    plt.ylabel("Force (N)", fontdict={'fontsize': 10, 'fontweight': 'bold', 'family': 'sans-serif'})

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12, loc='best')
    
    plt.grid()
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def do_raw_plots(data, show, plotprefix):
    count = 0
    for point, curve in data.items():
        s, i, j = point

        slope, baseline, selected_d, selected_f, selected_points, recorded = estimate_slope(curve)
        
        if plotprefix is not None:
            fname = f'{plotprefix}-{s:01d}-{i:03d}-{j:03d}.png'
            raw_plot_with_fit(point, curve, slope, baseline, selected_d, selected_f, selected_points, recorded, show=show, save=fname)
        
        print(f"{s} {i:03d} {j:03d} {slope:.5f}")
        count += 1

    print(f"# processing {count} spectra...")

def main(args):
    fname = args.textfile
    print(f"# parsing {fname}...")
    full_data = extract_data_points(fname)
    if args.first is not None:
        data = dict((k, v) for k, v in islice(full_data.items(), args.first))
    else:
        data = full_data
    do_raw_plots(data, args.show, args.plotprefix)

def get_argument_parser():
    p = ArgumentParser()
    p.add_argument("--textfile", "-t", required=True,
                   help="name of the data file containing AFM curves for many points")
    p.add_argument("--first", type=int,
                   help="number of curves to extract and plot")
    p.add_argument("--plotprefix", default="curve",
                   help="non-empty path prefix of plot files (PNGs); do not save plots if not given")
    p.add_argument("--show", action="store_true",
                   help="show each plot")
    return p

if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)

