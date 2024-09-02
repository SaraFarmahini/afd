#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from itertools import islice
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def extract_data_points(fname):
    
    # return a dict called 'data', such that
    # data[s, i, j] = (d, f),
    # where s is the series, i, j are the point coordinates;
    # d is a numpy array containing measured distances,
    # f is a numpy array containing measured forces.
    
    
    
    
    
    data = dict()
    with open(fname, 'rt') as ftext:
        # TODO: extract s, i, j, distance (d) and force (f) measurements from the data
        lines = ftext.readlines()
        
        points_num = Counter()
        
        i = None
        j = None
        s = None
        
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
                    data[(s, i, j)] = (d, f)  # this may have to be within some loop...
                    
                    #reseting distances and forces
                    distances.clear()
                    forces.clear()
                    
                    points_num[temp_indices] += 1
                    
                    
                    
    return data




def raw_plot(point, curve, save=None, show=True):
    """plot one raw distance-force curve"""
    # point is the triple (s, i, j) with series s, iIndex i, jIndex j
    # curve is the pair (d, f) of two numpy arrays with distances and forces
    s, i, j = point
    d, f = curve
    
    
    plt.figure(figsize=[9, 6])
    # TODO: do an actually nice plot here with title, axis labels, legend, etc
    plt.plot(d, f, color='g', lw=1.8, label=f'Series {s}, Point ({s}, {i}, {j})')  
    
    plt.title(f'Distance-Force Interaction {s}, Point ({s}, {i}, {j})', fontdict={'fontsize': 20, 'fontweight': 'bold', 'family': 'serif'})
    plt.xlabel("Distance (m)", fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
    plt.ylabel("Force (N)", fontdict={'fontsize': 15, 'fontweight': 'bold', 'family': 'serif'})
    
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
    for point, curve in data.items():
        s, i, j = point
        print(f"plotting curve at {point}")
        fname = f'{plotprefix}-{s:01d}-{i:03d}-{j:03d}.png' if plotprefix is not None else None
        raw_plot(point, curve, show=show, save=fname)


def main(args):
    fname = args.textfile
    print(f"parsing {fname}...")
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


# In[ ]:




