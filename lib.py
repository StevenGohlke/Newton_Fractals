import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

'''
    This package is to be used as a library. Please do not edit.
'''


def fpoly(x: np.float) -> np.float:
    """ Simple polynomial of degree 5"""
    return 0.009 * (x ** 5) + 0.02 * (x ** 4) - 0.32 * (x ** 3) - 0.54 * (x ** 2) + 3.2 * x - 1.0


def dfpoly(x: np.float) -> np.float:
    """Derivative of simple polynomial of degree 5"""
    return 5.0 * 0.009 * (x ** 4) + 4.0 * 0.02 * (x ** 3) - 3.0 * 0.32 * (x ** 2) - 2.0 * 0.54 * x + 3.2


# ======================================================================
# Functions for Newton Fractal

def generate_sampling(borders: list, size: int) -> np.ndarray:
    size_x = size
    size_y = int(size * (borders[3] - borders[2]) / (borders[1] - borders[0]))
    sx = np.linspace(borders[0], borders[1], size_x)
    sy = np.linspace(borders[2], borders[3], size_y)
    x, y = np.meshgrid(sx, sy)
    sampling = x + 1j * y
    return sampling, size_x, size_y


def get_colors(roots: np.ndarray) -> np.ndarray:
    colors = np.zeros((roots.shape[0], 3))
    c_idx = np.linspace(0.0, 1.0, roots.shape[0])
    cm = matplotlib.cm.get_cmap('jet')
    for idx, i in enumerate(c_idx):
        colors[idx] = cm(i)[:3]
    return colors



# Roots of unity
def rou(k):
    def f(x):
        return x ** k - 1

    return f


def drou(k):
    def f(x):
        return k * x ** (k - 1)

    return f


def rou_roots(k):
    return np.array([np.exp(2.j * np.pi * i / k) for i in range(k)])


rou_borders = [-1.5, 1.5, -1.5, 1.5]


# Polynomial
def poly(x):
    return x ** 3 - 2 * x + 2


def dpoly(x):
    return 3 * x ** 2 - 2


poly_roots = np.array([np.complex128(-1.76929235423863), np.complex128(0.884646177119316 + 0.589742805022206j),
                       np.complex128(0.884646177119316 - 0.589742805022206j)])
poly_borders = [-1.5, 0.5, -1.0, 1.0]


# Sinus function
def sin(x):
    return np.sin(x)


def dsin(x):
    return np.cos(x)


sin_roots = np.array(np.linspace(-10 * np.pi, 10 * np.pi, 21))
sin_borders = [-np.pi, np.pi, -np.pi, np.pi]

fractal_functions = [[rou(4), drou(4), rou_roots(4), rou_borders, "roots_of_unity_4"],
                     [rou(7), drou(7), rou_roots(7), rou_borders, "roots_of_unity_7"],
                     [poly, dpoly, poly_roots, poly_borders, "polynomial"],
                     [sin, dsin, sin_roots, sin_borders, "sinus"]]
