# fftmap
## About
This is a 2D map generation library for Python. You can use this to randomly generate different patterns in two dimensions. The features include:
- **Easy to use**; All the required functionality is implemented as a single python class.
- **On demand generation**; The map is generated as needed, allowing very large maps to be declared.
- **Repeatibility**; With any specific input parameters and seed, the generated map is always the same.
- **Customizable spatial frequency spectrum**; The fast Fourier transform (FFT) based implementation allows the use of a custom spectral filter.
- **Predictable statistics of output**; The output values tend to form a Gaussian distribution with zero mean and variance of one.

This library can be useful for example for:
- **Games**; A large randomly generated map is fun to explore.
- **Simulation**; Some real world phenomena can be simulated as 2D noise.
- **Patterns, art, etc.**; The output can be easily processed to generate camouflage patterns.

## Installation
The fftmap library is "installed" simply by dropping the "fftmap.py" file into suitable location and then importing it in Python. Numpy and scipy are needed to use this library.
 
## Examples
The use of the fftmap library is demonstrated by two example programs: fftmap_example1.py and fftmap_example2.py. Running these require additional libraries to be installed.
### Example 1 (fftmap_example1.py)
A simple geography is generated by creating random variation in the environmental conditions over the 2D plane. Total of three environmental variables (elevation, temperature, precipitation) are defined as fftmap.FFTMap objects. A color image is then rendered by polling the created objects for environmental data over a small region of the map. The color of each pixel is determined by the environmental conditions, for example places with low elevation are assumed to be lakes (blue color), and hot areas are assumed to be desert (deserty-red color). Different spatial frequency spectra are used for the three variables which causes the temperature to have much larger "structures" compared to the precipitation, which has smaller scale variation.

The output should look like this:

![Alt text](fftmap_example1_out.png?raw=true "title text")


### Example 2 (fftmap_example2.py)
This example demonstrates the statistical properties of values returned by fftmap.FFTMap objects. The distribution of returned values is a Gaussian distribution with zero mean and variance of one. This makes it easy to create a "2D map of boolean values" which is true for any desired percentage of pixels. In this example, 5% of the map is painted red, 5% green, and 10% blue. Different spatial frequency cutoffs are used to control the size of individual spots.

The output should look like this:

![Alt text](fftmap_example2_out.png?raw=true "title text")
