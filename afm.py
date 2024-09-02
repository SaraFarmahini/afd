"""
1. Rename this file to app.py
2. Download the preprocessed data files 'afm.heights.npy', 'afm.data.pickled' at:
   https://kingsx.cs.uni-saarland.de/index.php/s/KFrpMwCfJtaLpX3
3. streamlit run app.py -- afm
   This will take time to load the data and ESTIMATE the slopes with your code
   to be written here in function estimate_slope().
   Make sure that your code is efficient enough to run in a few seconds.
   Otherwise your application will seem to load forever.
"""

from argparse import ArgumentParser
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d  # reference: https://stackoverflow.com/questions/32155488/smoothing-a-2d-array-along-only-one-axis
import plotly.express as px  # mamba install plotly
# from streamlit_plotly_events import plotly_events # pip install streamlit_plotly_events



@st.cache_data
def load_data(prefix):
    print(f"- Loading {prefix}{{.data.pickled,.heights.npy}}")
    hname = f"{prefix}.heights.npy"
    H = np.load(hname)
    m, n = H.shape
    fname = f"{prefix}.data.pickled"
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Estimate all slopes
    slope_est = dict()  # maps tuple (s, i, j) to tuple (slope, anchors, info)
    for point, curve in data.items():
        s, i, j = point
        slope_est[point] = estimate_slope(curve, s)
    nseries = 1 + max(s for s, i, j in data.keys())
    slope_heatmaps = []
    for s in range(nseries):
        M = np.array([[(slope_est[s, i, j][0]) for j in range(n)] for i in range(m)], dtype=np.float64)
        slope_heatmaps.append(M)
    return H, data, slope_est, slope_heatmaps


def do_plot(point, curve, slope=None, anchors=None):
    """plot one distance-force curve with estimated slope"""
    s, i, j = point
    d, f = curve
    fig = plt.figure(figsize=[10, 6])
    plt.xlabel("distance (m)")
    plt.ylabel("force (N)")
    mode = 'push' if s == 0 else 'retract'
    plt.title(f"{mode} at ({i}, {j});  number of records: {len(d)}")
    label = f'data: {mode} at {(i, j)}'
    plt.scatter(d, f, s=1, label=label)
    plt.grid()
    if slope is not None and anchors is not None:
        anchor0, anchor1 = anchors[0], anchors[1]
        plt.axline(anchor0, slope=slope, color='red', linestyle='--', label=f'{slope:.4g} N/m')
        plt.plot([anchor0[0]], [anchor0[1]], 'rx')
        plt.plot([anchor1[0]], [anchor1[1]], 'rx')
        
    # if flat_start is not None:
    #     flat_start_d, flat_start_f = flat_start
    #     if flat_start_d is not None and flat_start_f is not None:
    #         ax.plot(flat_start_d, flat_start_f, 'g+', markersize=12, label='Flat Start')
    
    
    plt.legend()
    return fig



def flat_side(data, wnd=30, h_tolerance=0.002):  # For series 0
    d, f = data
    
    #In the following line, I used uniform_filter1d from SciPy library to smooth the noisy flat part of the plot
    #my goal was to find the start point of the flat part and then find the slope right before the flatpart for series 0 plots
    #without using this function (to smooth flat part) my code find the first noises as the slope!
    
    
    smoothed_f = uniform_filter1d(f, size=wnd)  
    # reference: https://stackoverflow.com/questions/32155488/smoothing-a-2d-array-along-only-one-axis
    
    
    flat_part = []
    
    
    for i in range(len(smoothed_f) - 1, 0, -1):  # moving from last point (right to left) to capture last flat part of the plot
        delta_f = smoothed_f[i] - smoothed_f[i - 1]
        delta_d = d[i] - d[i - 1]
        slope = abs(delta_f / delta_d)
        if slope > h_tolerance:
            flat_point = i + 1
            flat_part.append(flat_point)
            
    flat_start = flat_part[0]  # defining the first point of the flat part to estimate the slope before this part
    
    
    
    return flat_start, smoothed_f




def longest_descending(data, tolerance=3):
    # For series=1, so if you're only interested in my algorithm series = 0, please ignore it and go to the next function.
    #I used my previous code for series1, since the new algorithm does not define the wanted slope for series 1.
    
    d, f = data
    longest_segment_d = []
    longest_segment_f = []
    max_length = 0

    start = 0
    while start < len(d) - 1:
        segment_length = 0
        fluctuation = 0
        current = start

        while current < len(d) - 1:
            if f[current] > f[current + 1]:
                fluctuation = 0
            else:
                fluctuation += 1
            
            segment_length += 1

            if fluctuation >= tolerance:
                break

            current += 1

        if segment_length > max_length:
            max_length = segment_length
            longest_segment_d = d[start:start + segment_length]
            longest_segment_f = f[start:start + segment_length]

        start += max(segment_length, 1)

    longest_segment = (longest_segment_d, longest_segment_f)
    
    return longest_segment
    


def estimate_slope(curve, s, wnd=50, s_tolerance=0.0013):
    d, f = curve
    if s == 0:
        d, f = d[::-1], f[::-1]  # reverse d and f for series-0 spectra!
        flat_start, smoothed_f = flat_side((d, f), wnd, s_tolerance)
        
        
        
        if flat_start == 0:
            raise ValueError("Not detectable descending part")
        
        
        
        selected_d = d[:flat_start]
        selected_f = f[:flat_start]
        
        
        # In the following part I wanted to make sure the anchors points are close enough to the flat_start point as the professor said the line should be fixed to the lower part of the descending line
        # Numbers (32 , 70) has been obtaining by testing on diferent points 
        anchor1_ix = max(0, len(selected_f) - 32)
        anchor2_ix = max(0, len(selected_f) - 70)
        
        
        anchor1_d = selected_d[anchor1_ix]
        anchor1_f = selected_f[anchor1_ix]
        anchor2_d = selected_d[anchor2_ix]
        anchor2_f = selected_f[anchor2_ix]
        
        
        result = linregress([anchor1_d, anchor2_d], [anchor1_f, anchor2_f])
        slope = result.slope
        #reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        
        intercept = result.intercept
        anchor1_y = intercept + slope * anchor1_d
        anchor2_y = intercept + slope * anchor2_d
        anchor1 = (anchor1_d, anchor1_y)
        anchor2 = (anchor2_d, anchor2_y)
        anchors = (anchor1, anchor2)
        
        
        info = None  # can be anything you want to return in addition
        return slope, anchors, info



    elif s == 1:
        
        longest_seg_d, longest_seg_f = longest_descending((d, f))

        if longest_seg_d.size == 0 or longest_seg_f.size == 0:
            raise ValueError("Not detectable descending part")
     
        
        anchor1_ix = 0
        anchor2_ix = len(longest_seg_d) - 1
        
        
        anchor1_d = longest_seg_d[anchor1_ix]
        anchor1_f = longest_seg_f[anchor1_ix]
        anchor2_d = longest_seg_d[anchor2_ix]
        anchor2_f = longest_seg_f[anchor2_ix]
        
        
        result = linregress([anchor1_d, anchor2_d], [anchor1_f, anchor2_f])
        slope = result.slope
        intercept = result.intercept
        
        anchor1_y = intercept + slope * anchor1_d
        anchor2_y = intercept + slope * anchor2_d
        anchor1 = (anchor1_d, anchor1_y)
        anchor2 = (anchor2_d, anchor2_y)
        anchors = (anchor1, anchor2)
        
        
        
        info = None  # can be anything you want to return in addition
        return slope, anchors, info


# MAIN script

p = ArgumentParser()
p.add_argument("prefix",
    help="common path prefix for spectra (.data.pickled) and heights (.heights.npy)")
args = p.parse_args()
prefix = args.prefix

st.sidebar.title("AFM Data Explorer")
st.sidebar.write(f"Path prefix:\n'{prefix}'")
H, S, slope_est, slope_heatmaps = load_data(prefix)  # cached
m, n = H.shape
nseries = len(slope_heatmaps)


selected_series = st.sidebar.selectbox(
    "Series:",
    range(nseries),
    index=0
)

selected_i = st.sidebar.slider(
    "Coordinate i (vertical):",
    min_value=0,
    max_value=m - 1,
    value=0
)

selected_j = st.sidebar.slider(
    "Coordinate j (horizontal):",
    min_value=0,
    max_value=n - 1,
    value=0
)

# reference: https://snyk.io/advisor/python/streamlit/functions/streamlit.sidebar.slider

st.markdown("## Heights")
fig_h = px.imshow(H, color_continuous_scale='Turbo')
fig_h.update_layout(width=700, height=500)
st.plotly_chart(fig_h)
st.markdown("## Slopes")
fig_s = px.imshow(slope_heatmaps[selected_series], color_continuous_scale='Turbo')
fig_s.update_layout(width=700, height=500)
st.plotly_chart(fig_s)



point = (selected_series, selected_i, selected_j)
curve = S[point]
slope, anchors, flat_start = slope_est[point]
fig = do_plot(point, curve, slope, anchors)
st.pyplot(fig)

# Reference: https://www.restack.io/docs/streamlit-knowledge-streamlit-chart-size-guide




# All used references:
# 1. https://stackoverflow.com/questions/32155488/smoothing-a-2d-array-along-only-one-axis
# 2. https://snyk.io/advisor/python/streamlit/functions/streamlit.sidebar.slider
# 3. https://www.restack.io/docs/streamlit-knowledge-streamlit-chart-size-guide
#4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html