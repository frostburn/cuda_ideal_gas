from pylab import *
from matplotlib.animation import FuncAnimation

import numpy as np

from subprocess import Popen, PIPE, check_output

ROUNDS = 10000
DIMS = 2
RADIUS = 0.003
SIZEOF_FLOAT = 4
DENSITY = 32
GRID_WIDTH = 31

check_output([
    'nvcc', 'engine.cu',
    '-DROUNDS={}'.format(ROUNDS),
    '-DDIMS={}'.format(DIMS),
    '-DDENSITY={}'.format(DENSITY),
    '-DGRID_WIDTH={}'.format(GRID_WIDTH),
    '-DRADIUS={}'.format(RADIUS),
])

fig, ax = subplots()
ax.set_xlim((0, GRID_WIDTH))
ax.set_ylim((0, GRID_WIDTH))

with Popen('./a.out', stdout=PIPE) as p:
    res = p.stdout.read(DIMS*SIZEOF_FLOAT*DENSITY*GRID_WIDTH**2)
    pos = np.frombuffer(res, dtype='float32').reshape(DENSITY*GRID_WIDTH**2, DIMS)

    plt = plot(pos[:, 0], pos[:, 1], "b,")

    def init():
        return plt

    def update(frame):
            res = p.stdout.read(DIMS*SIZEOF_FLOAT*DENSITY*GRID_WIDTH**2)
            pos = np.frombuffer(res, dtype='float32').reshape(DENSITY*GRID_WIDTH**2, DIMS)

            plt[0].set_data(pos[:, 0], pos[:, 1])
            return plt

    ani = FuncAnimation(fig, update, frames=range(ROUNDS - 1), init_func=init, blit=True, repeat=False, interval=0)
    show()
