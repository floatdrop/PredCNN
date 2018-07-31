import config
import numpy as np

shape = (10000, 20, config.input_shape[0], config.input_shape[1], 1)

file = np.memmap('data.dat', shape=shape, mode='w+', dtype=np.float32)

file[:] = np.random.rand(*shape)

del file
