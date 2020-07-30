"""
fftmap_example1.py - example of fftmap library use

This program demonstrates the use of fftmap library.

--------------------------------------------------------------------------------
LICENCE - MIT Licence

Copyright (c) 2020 Markku Alamäki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------------
"""

import fftmap
import math
import random
import PIL.Image
import PIL.ImageDraw

# Map size
SIZE = 10000
RANGES = [[-SIZE, SIZE], [-SIZE, SIZE]]

# Function for generating a spctral filter with a cutoff wavelength
def get_spectral_filter(wave_cutoff):
    # Small number to avoid divide by zero error
    epsilon = 0.000000001
    # Spectral filter
    #     - (1/k)^2 factor prevents the high frequencies from dominating too much
    #     - A short pass step filter removes longest wavelengths
    return lambda k: int((1 / (k + epsilon)) < wave_cutoff) * 1 / (k + epsilon) ** 2

# Function for creating maps with specified spatial frequency cutoff
def create_map(wave_cutoff, seed=None):
    # Use large enough block size to support maximum wavelength
    block_size = int(wave_cutoff)

    # get filter
    spectral_filter = get_spectral_filter(wave_cutoff)

    # We select the numpy's ndarray since it gives less overhead with the
    # FFTMap.get_value method.
    array_type = "ndarray"

    # Create the map and return it
    return fftmap.FFTMap(RANGES, block_size, spectral_filter, seed=seed, array_type=array_type)


# Create maps of different environmental data
temperature_map = create_map(300, seed=0)
elevation_map = create_map(100, seed=1)
precipitation_map = create_map(50, seed=2)

# Map the values to something more tangible
# Temperature is average of 10degC, with standard deviation 15degC
temperature_fun = lambda x, y: 10 + 15 * temperature_map.get_value(x, y)
# Elevation is average of 50m, with standard deviation 100m
elevation_fun = lambda x, y: 50 + 100 * elevation_map.get_value(x, y)
# Precipitation in mm/day
precipitation_fun = lambda x, y: max(0, 3.0 + (precipitation_map.get_value(x, y)))

# Create a function to calculate pixel color from environmental conditions
def color_fun(x, y):
    # Fetch environmental data
    elevation = elevation_fun(x, y)
    temperature = temperature_fun(x, y)
    precipitation = precipitation_fun(x, y)

    # If the elevation is below zero, assume that the location is sea
    if elevation < 0:
        # Return blue color
        return (20, 20, 100)

    # Depending on temperature, select base color
    if temperature > 30:
        # Hot areas are desert, with deserty color
        color = (150, 100, 70)
    elif temperature > 5:
        # Green color for temperate regions
        color = (100, 150, 100)
    else:
        # Snow white color for the coldest areas
        color = (200, 200, 200)

    # Depending on rain, apply vegetation, which is denoted by slightly darker color
    if 2 + 2 * random.random() < precipitation:
        color = tuple(color[i] // 2 for i in range(3))

    return color

# Render images of the map
print("Rendering image...")
IMAGE_SIZE_X = 500
IMAGE_SIZE_Y = 500
CONTOURS = 4
image = PIL.Image.new("RGB", [IMAGE_SIZE_X * 2, IMAGE_SIZE_Y * 2])
for x in range(IMAGE_SIZE_X):
    # Track progress
    if x % int(IMAGE_SIZE_X / 20) == 0:
        print("    %d%% done" % (100 * x / IMAGE_SIZE_X))
    # Inner loop
    for y in range(IMAGE_SIZE_Y):
        # Render main image to the upper left area of image
        color = color_fun(x, y)
        image.putpixel((x, y), color)

        # Render elevation map to upper right area of image
        value = elevation_map.get_value(x, y)
        color = int(value * CONTOURS / 2) * (127 // CONTOURS) + 127
        image.putpixel((IMAGE_SIZE_X + x, y), tuple(color for i in range(3)))

        # Render temperature...
        value = temperature_map.get_value(x, y)
        color = int(value * CONTOURS / 2) * (127 // CONTOURS) + 127
        image.putpixel((x, IMAGE_SIZE_Y + y), tuple(color for i in range(3)))

        # Render precipitation...
        value = precipitation_map.get_value(x, y)
        color = int(value * CONTOURS / 2) * (127 // CONTOURS) + 127
        image.putpixel((IMAGE_SIZE_X + x, IMAGE_SIZE_Y + y), tuple(color for i in range(3)))

image_draw = PIL.ImageDraw.Draw(image)
image_draw.text((IMAGE_SIZE_X + 10, 10), "ELEVATION", fill=(255, 0, 0))
image_draw.text((10, IMAGE_SIZE_Y + 10), "TEMPERATURE", fill=(255, 0, 0))
image_draw.text((IMAGE_SIZE_X + 10, IMAGE_SIZE_Y + 10), "PRECIPITATION", fill=(255, 0, 0))
image.save("fftmap_example1_out.png")
