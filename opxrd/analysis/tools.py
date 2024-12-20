import sys

import numpy as np
from IPython.core.display_functions import display
from mistune import Markdown
from numpy.typing import NDArray


def print_text(msg: str):
    if 'ipykernel' in sys.modules:
        display(Markdown(msg))
    else:
        print(msg)


def compute_fourier(x: NDArray, y: NDArray):
    N = len(y)
    T = (x[-1] - x[0]) / (N - 1)

    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])
    return xf, yf