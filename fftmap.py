""" 
fftmap library:
    
    This library can be used to create 2D maps. Map here means a large
    "virtual" array of floating point numbers, which is virtual in the sense
    that the array is filled procedurally, block by block, as needed. The map
    can be defined to contain different spatial frequencies in different 
    proportions. 

    The block by block generation of the map is achieved by allocating arrays as
    needed, and then filling those arrays with random data. The random data is
    first generated as randomized frequency spectrum, which is then filtered
    according to user defined spectral weighting. This weighted spectrum is then
    transformed into "position space" data via 2D FFT. This position space
    data is in turn filtered with a window function to generate a noise block
    with smoothly decaying edges. Multiple noise blocks obtained in this way 
    are then interlaced together to obtain smooth noise spectrum everywhere in
    the map.

    For any set of input parameters, including seed, the generated map is always
    the same, regardless of the order in which the map is created or explored.
    
    It is possible to declare very large maps without using much memory, since
    the arrays are only allocated when needed. Especially if the FFTMap object
    is made to use sparse arrays for some data structures, the maps can be
    made practically unlimited in size.

    See the documentation on the FFTMap class for details of how to use.

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
import numpy as np
import scipy.sparse

def spline(x):
    if x < -0.5:
        return 4 * (0.5 * x ** 2 + x + 0.5)
    if x < 0.5:
        return 4 * (-0.5* x ** 2 + 0.25)
    else:
        return 4 * (0.5 * x ** 2 - x + 0.5)

def create_filter_window(block_size):
    window = np.zeros([block_size])
    for i in range(block_size):
        window[i] = spline((2 * i - (block_size - 1)) / block_size)
    return window

class FFTMap_:
    """
        This is the "core" class, which is not intended to be used directly.
        Use FFTMap instead.
    """
    def __init__(self, n_blocks, block_size, spectral_filter, seed=None, array_type="ndarray"):
        self.n_blocks_x = n_blocks[0]
        self.n_blocks_y = n_blocks[1]
        self.block_size = block_size
        self.fft_block_size = 2 * block_size
        
        # Completed blocks are added here
        self.blocks_list = []

        if array_type == "ndarray":
            # Indices to self.blocks_list
            self.blocks_indices = np.full([self.n_blocks_x, self.n_blocks_y], -1)
            # Status of blocks
            self.blocks_initialized = np.full([self.n_blocks_x, self.n_blocks_y], False)
            self.blocks_finished = np.full([self.n_blocks_x, self.n_blocks_y], False)
            self.fft_blocks_applied = np.full([self.n_blocks_x, self.n_blocks_y], False)
        elif array_type == "dok_matrix":
            # Indices to self.blocks_list
            self.blocks_indices = scipy.sparse.dok_matrix((self.n_blocks_x, self.n_blocks_y), dtype=int)
            # Status of blocks
            self.blocks_initialized = scipy.sparse.dok_matrix((self.n_blocks_x, self.n_blocks_y), dtype=bool)
            self.blocks_finished = scipy.sparse.dok_matrix((self.n_blocks_x, self.n_blocks_y), dtype=bool)
            self.fft_blocks_applied = scipy.sparse.dok_matrix((self.n_blocks_x, self.n_blocks_y), dtype=bool)
        else:
            raise ValueError("Array type not recognized.")

        # Create masks
        self.calculate_spectral_mask(spectral_filter)
        self.calculate_spatial_mask()

        # Normalize spectral mask
        self.normalize_spectral_mask()

        # Seed
        if type(seed) == type(None):
            self.seed = np.random.PCG64().random_raw()
        elif type(seed) == int:
            self.seed = seed
        else:
            raise ValueError("Invalid seed. Must be an integer.")

    # Get a single value
    def get_value(self, x, y):
        # Calculate block index and extra offset
        block_index_x, extra_x = divmod(x, self.block_size)
        block_index_y, extra_y = divmod(y, self.block_size)

        # Check input
        if (block_index_x < 1) or (block_index_x > self.n_blocks_x - 2):
            raise ValueError("block_index_x=%d is out of bounds" % block_index_x)
        if (block_index_y < 1) or (block_index_y > self.n_blocks_y - 2):
            raise ValueError("block_index_y=%d is out of bounds" % block_index_x)

        # Finish creating the target block
        if not self.blocks_finished[block_index_x, block_index_y]:
            self.finish_block(block_index_x, block_index_y)

        # Retrieve the value
        block = self.blocks_list[self.blocks_indices[block_index_x, block_index_y]]
        return block[extra_x, extra_y]

    # Get a range of values. This has a lower overhead compared to doing the
    # same thing with get_value.
    def get_values(self, x1, x2, y1, y2):
        # Calculate the block indices, and extra offsets
        block_index_x1, extra_x1 = divmod(x1, self.block_size)
        block_index_x2, extra_x2 = divmod(x2, self.block_size)
        block_index_y1, extra_y1 = divmod(y1, self.block_size)
        block_index_y2, extra_y2 = divmod(y2, self.block_size)
        
        # Check input
        if (block_index_x1 < 1):
            raise ValueError("block_index_x1=%d is out of bounds" % block_index_x1)
        if (block_index_x2 > self.n_blocks_x - 2):
            raise ValueError("block_index_x2=%d is out of bounds" % block_index_x2)
        if (block_index_y1 < 1):
            raise ValueError("block_index_y1=%d is out of bounds" % block_index_y1)
        if (block_index_y2 > self.n_blocks_y - 2):
            raise ValueError("block_index_y2=%d is out of bounds" % block_index_y2)
        if (x1 >= x2) or (y1 >= y2):
            raise ValueError("Invalid range specified for get_values.")

        # Finish creating the required blocks
        for block_index_x in range(block_index_x1, block_index_x2 + 1):
            for block_index_y in range(block_index_y1, block_index_y2 + 1):
                if not self.blocks_finished[block_index_x, block_index_y]:
                    self.finish_block(block_index_x, block_index_y)

        # Retrieve the values
        copied_data = np.empty([x2 - x1, y2 - y1], dtype=float)
        x_data_size_total = x2 - x1
        y_data_size_total = y2 - y1
        
        copied_data_x_index = 0
        block_index_x = block_index_x1
        while copied_data_x_index < x_data_size_total:
            # Prepare
            if block_index_x == block_index_x1:
                blx1 = extra_x1
            else:
                blx1 = 0
            if block_index_x == block_index_x2:
                blx2 = extra_x2
            else:
                blx2 = self.block_size

            x_data_size_step = blx2 - blx1
            retx1 = copied_data_x_index
            retx2 = copied_data_x_index + x_data_size_step

            # Copy
            copied_data_y_index = 0
            block_index_y = block_index_y1
            while copied_data_y_index < y_data_size_total:
                # Prepare
                if block_index_y == block_index_y1:
                    bly1 = extra_y1
                else:
                    bly1 = 0
                if block_index_y == block_index_y2:
                    bly2 = extra_y2
                else:
                    bly2 = self.block_size

                y_data_size_step = bly2 - bly1
                rety1 = copied_data_y_index
                rety2 = copied_data_y_index + y_data_size_step

                # Copy
                block = self.blocks_list[self.blocks_indices[block_index_x, block_index_y]]
                copied_data[retx1:retx2, rety1:rety2] = block[blx1: blx2, bly1: bly2]

                # Increment
                copied_data_y_index += y_data_size_step
                block_index_y += 1
            # Increment
            copied_data_x_index += x_data_size_step
            block_index_x += 1

        return copied_data


    def finish_block(self, block_index_x, block_index_y):
        # Apply fft blocks
        for i in (-1, 0):
            for j in (-1, 0):
                fft_index_x = block_index_x + i
                fft_index_y = block_index_y + j
                if not self.fft_blocks_applied[fft_index_x, fft_index_y]:
                    self.apply_fft_block(fft_index_x, fft_index_y)

        # Mark this block as completed
        self.blocks_finished[block_index_x, block_index_y] = True

    def apply_fft_block(self, fft_index_x, fft_index_y):
        # Initialize the normal blocks in this fft block's area of effect
        for i in (0, 1):
            for j in (0, 1):
                block_index_x = fft_index_x + i
                block_index_y = fft_index_y + j
                if not self.blocks_initialized[block_index_x, block_index_y]:
                    block = np.zeros([self.block_size, self.block_size])
                    block_list_index = len(self.blocks_list)
                    self.blocks_list.append(block)
                    self.blocks_initialized[block_index_x, block_index_y] = True
                    self.blocks_indices[block_index_x, block_index_y] = block_list_index

        # Create this fft block
        # Get the random noise block for this fft block
        temp_array = self.get_rng_mask(fft_index_x, fft_index_y)
        # Apply the spectral filter mask
        temp_array *= self.spectral_mask
        # Transform the filtered noise to waves
        temp_array = np.fft.fft2(temp_array)
        # Take the real part of noise
        temp_array = temp_array.real
        # Apply the spatial filter mask
        temp_array *= self.spatial_mask
        
        # Apply this fft block to any blocks necessary
        for i in (0, 1):
            for j in (0, 1):
                self.blocks_list[
                    self.blocks_indices[fft_index_x + i, fft_index_y + j]
                ] += (
                    temp_array[
                        i * self.block_size : (1 + i) * self.block_size,
                        j * self.block_size : (1 + j) * self.block_size
                    ]
                )

        # Mark this fft block as applied
        self.fft_blocks_applied[fft_index_x, fft_index_y] = True

    # Creates spectral filtering mask
    def calculate_spectral_mask(self, spectral_filter):
        spectral_mask = np.zeros([self.fft_block_size, self.fft_block_size])
        for i in range(self.fft_block_size):
            for j in range(self.fft_block_size):
                i_ = self.fft_block_size - i
                j_ = self.fft_block_size - j
                spectral_mask[i, j] = (
                    spectral_filter(np.sqrt(i ** 2 + j ** 2) / self.fft_block_size) +
                    spectral_filter(np.sqrt(i_ ** 2 + j ** 2) / self.fft_block_size) +
                    spectral_filter(np.sqrt(i ** 2 + j_ ** 2) / self.fft_block_size) +
                    spectral_filter(np.sqrt(i_ ** 2 + j_ ** 2) / self.fft_block_size)
                )
        self.spectral_mask = spectral_mask

    # Creates spatial filtering mask for interlacing noise blocks together
    def calculate_spatial_mask(self):
        # We start with a 1 dimensional tapering window function, which has the
        # property that multiple displaced copies of them can be summed together
        # to yield a constant value everywhere.
        filter_window = create_filter_window(self.fft_block_size)
        #filter_window = np.hanning(self.fft_block_size)

        # However, because we will not be summing together the windows directly,
        # but instead noise blocks scaled by the window function, we need to
        # take the square root of the window to get even variance everywhere
        # after summing. This is because summing two noises with random relative
        # phase do not directly add up in average amplitude or standard
        # deviation, but instead the variances do add up.
        filter_window_sqrt = np.sqrt(filter_window)

        # Finally the mask is made two dimensional by taking the outer product
        # with itself.
        self.spatial_mask = filter_window_sqrt.reshape(self.fft_block_size, 1) * filter_window_sqrt

    # Creates random mask which is repeateble for the map
    def get_rng_mask(self, fft_index_x, fft_index_y):
        # Create rng
        random_numbers_per_block = 2 * self.fft_block_size ** 2
        block_index = (fft_index_x * self.n_blocks_y + fft_index_y)
        advance_amount = block_index * random_numbers_per_block
        seed_seq = np.random.SeedSequence(self.seed).spawn(1)[0]
        pcg = np.random.PCG64(seed_seq).advance(advance_amount)
        rng = np.random.Generator(pcg)
        
        # Create mask
        # Base array with complex type
        temp_array = np.zeros([self.fft_block_size, self.fft_block_size], dtype=np.complex128)
        # Amplitude from uniform distribution
        temp_array += rng.random([self.fft_block_size, self.fft_block_size])
        # Phase
        temp_array *= np.exp(2j * np.pi * rng.random([self.fft_block_size, self.fft_block_size]))
        return temp_array

    # Normalizes the spectral mask. The normalization matches with the random 
    # mask returned by get_rng_mask in a special way. Specifically, if we take
    # the matrix returned by get_rng_mask and the normalized spectral mask, and
    # take the Hadamard product between them, we get a matrix whose variance
    # of the real part is on average one.
    def normalize_spectral_mask(self):
        spectral_mask_squared_sum = np.sqrt(np.sum(self.spectral_mask**2))
        if spectral_mask_squared_sum == 0:
            raise ValueError("Spectral mask is zero everywhere and can't be normalized.")
        normalization_factor = np.sqrt(6) / spectral_mask_squared_sum
        self.spectral_mask *= normalization_factor


# Main implementation
class FFTMap:
    def __init__(self, ranges, block_size, spectral_filter, seed=None, array_type="ndarray"):
        """
            - ranges: the desired size of the map. Give four integers in form
            [[x1, x2], [y1, y2]]. Then the allowed indices will be [x1, x2 - 1]
            and [y1, y2 - 1], i.e. x2 and y2 are excluded.
            - block_size: (integer) The map is generated in this sized chunks.
            Using a larger block_size allows lower frequencies to be present,
            and also minimizes the overhead of some operations.
            - spectral_filter: This determines the spatial frequency spectrum.
            It is a function that takes as argument a spatial frequency in
            cycles per pixel width, and returns a weighing factor for that
            frequency. The weighing factor should go to zero for low
            frequencies. Optimally the wavelength, i.e. inverse of spatial
            frequency should be smaller than the block size.
            - seed: Define a seed for the map.
            - array_type: Possible values "dok_matrix" and "ndarray", which mean
            either the normal numpy.ndarray or the dictionary of keys "DOK"
            matrix from scipy.sparse.dok_matrix.. Selecting the ndarray type
            allows less overhead with the get_value method than the DOK matrix.
            This overhead can be mitigated by using larger block size and
            fetching larger chunks of data with the get_values method.
            FFTMap will store the status of blocks into arrays. This includes
            information such as whether the block has been initialized or not,
            and the location of initialized blocks in a list. The size of these
            arrays may be very large, especially when the map is big and the
            block size is small. The DOK matrix can be used if _very_ large maps
            need to be declared, since it is efficient with storing sparce
            matrices. 

        """
        try:
            assert ranges[0][0] < ranges[0][1]
            assert ranges[1][0] < ranges[1][1]
        except:
            raise ValueError("Invalid ranges given.")

        self.x1, self.x2 = ranges[0]
        self.y1, self.y2 = ranges[1]
        try:
            assert type(self.x1) == int
            assert type(self.x2) == int
            assert type(self.y1) == int
            assert type(self.y2) == int
        except:
            raise ValueError("Invalid ranges given. Boundaries must be integers.")
        try:
            assert type(block_size) == int
            assert block_size > 0
        except:
            raise ValueError("Invalid block size. Block size must be a positive integer.")
        if not "__call__" in dir(spectral_filter):
            raise ValueError("Invalid spectral filter.")

        self.block_size = block_size

        required_usable_blocks_x = (self.x2 - self.x1) // block_size + 1
        required_usable_blocks_y = (self.y2 - self.y1) // block_size + 1

        required_blocks_x = required_usable_blocks_x + 2
        required_blocks_y = required_usable_blocks_y + 2

        self.mapping = FFTMap_([required_blocks_x, required_blocks_y], block_size, spectral_filter, seed, array_type)

    def get_value(self, x, y):
        """
            - Fetch the value at the given location.
            - The arguments x, y are integers from the ranges declared in the 
            constructor.
            - This method automatically generates new blocks as needed. Fetching
            values from a block that has not been accessed before will therefore
            take a longer time.
            - The distribution of returned values resembles a Gaussian
            distribution due to central limit theorem. The distribution has
            mean value of zero and variance of one.
            - For fetching larger chunks of data the get_values method is
            recommended instead, as it has much lower overhead.
        """
        if (x < self.x1) or (x >= self.x2):
            raise ValueError("x=%d is out of bounds!" % x)
        if (y < self.y1) or (y >= self.y2):
            raise ValueError("y=%d is out of bounds!" % y)

        return self.mapping.get_value(
            x - self.x1 + self.block_size,
            y - self.y1 + self.block_size
        )

    def get_values(self, x1, x2, y1, y2):
        """
            - A low overhead version of get_value, that fetches larger chunks
            of data in one call.
            - input is the x and y range of the data
            - returns a numpy.ndarray
        """
        return self.mapping.get_values(
            x1 - self.x1 + self.block_size,
            x2 - self.x1 + self.block_size,
            y1 - self.y1 + self.block_size,
            y2 - self.y1 + self.block_size
        )


