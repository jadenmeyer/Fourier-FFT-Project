# Fourier Analysis Project
![License](https://img.shields.io/badge/license-CC%20BY--NC-lightgrey)

Final project for Fourier Analysis. 
This project looks at the Fast Fourier Transform algorithim used in many fields of analysis. It will cover the analysis speeds of CPUs vs GPUs currently the matlab code is optimized for a GTX 1070 running on CUDA 6.1 (all I have at the moment) and a Ryzen 7 7700x.
IF YOU DO NOT HAVE AN NVIDIA GPU THE MATLAB CODE WILL NOT WORK. IF YOU HAVE AN NVDIA GPU YOU NEED AT LEAST 8 GB OF VRAM TO RUN IT.

If you are curious what the PGFplots folder is, that is just the tex file I use to test graphs and see if they work well with my document.

This will cover a very basic introduction to how CUDA goes about parallelizing tasks starting with CUDA Threads -> CUDA Blocks -> CUDA Grids. There is a basic introduction to what a streaming multiprocessor is. As it stands right now, there is a commit focused on SM and CUDA parallelization. In the coming days, there will be an explanation on what the FFT is later focusing on RADIX-2, Cooly-Tukey, and Butterly Operations. If you are curious there is a great video on MIT OpenCourseWare.

Later, there will include code and plots (in Matlab possibly some C if I have time) for the Heat Equation and such solutions. LaTeX documentation will exist for this.

If something looks off feel free to let me know, just know this is a work in progress!

Free to use information. The license states no corporate use of the information. Please respect that wish! 
