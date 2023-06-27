from numba import cuda
import numpy as np

an_array = np.empty(100)
threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
# increment_by_one[blockspergrid, threadsperblock](an_array)


if __name__ == "__main__":
    print(blockspergrid)