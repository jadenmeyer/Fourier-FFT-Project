import matplotlib.pyplot as plt
import matplotlib.pyplot as pt
import numpy as np

f = open("pycode\gpu_times_seconds.txt")
g = open("pycode\gpu_fft_times_seconds.txt")

DFTxpoints = []
DFTypoints = []
FFTxpoints = []
FFTypoints = []

for line in f:
        # Split the line into x and y values and convert to float
        parts = line.split()
        if len(parts) == 2:
            DFTxpoints.append(float(parts[0]))
            DFTypoints.append(float(parts[1]))

for line in g:
        # Split the line into x and y values and convert to float
        parts = line.split()
        if len(parts) == 2:
            FFTxpoints.append(float(parts[0]))
            FFTypoints.append(float(parts[1]))

f.close()
g.close()

plt.xlabel("Size")
plt.ylabel("Seconds")
plt.title("GPU DFT")
plt.plot(DFTxpoints, DFTypoints, 'o:b')
plt.show()

pt.xlabel("Size")
pt.ylabel("Seconds")
pt.title("GPU FFT")
pt.plot(FFTxpoints, FFTypoints, 'o:g')
pt.show()

