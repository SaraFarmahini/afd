
The goal is to analyze data from an atomic force microscope:  
One examines a flat surface with “blobs” of some unknown material on it.
A thin needle is pushed down onto the surface until it hits the surface or the blob.
The surface or the material then pushes back, which results in a measurable force. This is repeated on many locations (i,j) on the surface, i = 0,...,127, j = 0,...,127.
       
Part I  
Goal:  
The first goal is to extract the distance and force data from the text file
for each point (i,j) on the surface (in two distinct measurement series).
The second goal is to plot each measurement as a simple graph to examine the data

Part II  
Goal:  
We want to estimate the slope of the linear-looking piece of the curve on the left side of the series-0 plots (the first data series for each pixel).


Part III:  
Goal:  
The task is to write a streamlit app (filename: app.py) that loads the data and estimates the slope of all measurements by running streamlit run app.py -- afm (where afm is the filename prefix of the dataset),displays the height information as a heatmap, displays the estimated slopes at every pixel (for chosen s) as a heatmap, allows the user to select series s, vertcial coordinate i, horizontal coordinate j, displays the measurement series (distance vs. force, as before) at (s, i, j)An example is on the next slide. A code template (for loading data, etc.) is provided in the Materials section.
