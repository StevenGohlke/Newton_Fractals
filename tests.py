import numpy as np
import unittest
import matplotlib.pyplot as plt

from lib import fpoly, dfpoly, fractal_functions, generate_sampling, get_colors
from main import find_root_bisection, find_root_newton, generate_newton_fractal


def helper_function_generate_newton_fractal(function):
    size = 100# size of the image
    max_iterations = 200

    for c, el in enumerate(function):
        f, df, roots, borders, name = el
        sampling, size_x, size_y = generate_sampling(borders, size)
        res = generate_newton_fractal(f, df, roots, sampling, n_iters_max=max_iterations)
        colors = get_colors(roots)

        # Generate image
        img = np.zeros((sampling.shape[0], sampling.shape[1], 3))
        for i in range(size_y):
            for j in range(size_x):
                if res[i, j][1] <= max_iterations:
                    img[i, j] = colors[res[i, j][0]] / max(1.0, res[i, j][1] / 6.0)

        plt.imsave('data/fractal_' + name + '.png', img)
        return img


class Tests(unittest.TestCase):
    def test_1_find_root_bisection(self):
        x0 = find_root_bisection(lambda x: x ** 2 - 2, np.float64(-1.0), np.float64(2.0))
        self.assertTrue(np.isclose(x0, np.sqrt(2)))
        x1 = find_root_bisection(fpoly, np.float64(-1.0), np.float64(5.0))
        x2 = find_root_bisection(fpoly, np.float64(1.0), np.float64(4.0))
        x3 = find_root_bisection(fpoly, np.float64(4.0), np.float64(5.0))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x1, x2, x3], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()

    def test_2_find_root_newton(self):
        x0, i0 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(10.0))
        x1, i1 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(5.0))
        x2, i2 = find_root_newton(lambda x: x ** 2 - 2, lambda x: 2 * x, np.float64(0.1))
        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([np.sqrt(2)] * 3)))

        x0, i0 = find_root_newton(fpoly, dfpoly, np.float64(-1.0))
        x1, i1 = find_root_newton(fpoly, dfpoly, np.float64(2.0))
        x2, i2 = find_root_newton(fpoly, dfpoly, np.float64(5.0))
        self.assertTrue(np.allclose(np.array([x0, x1, x2]), np.array([0.335125152578, 2.61080833945, 4.79087461944])))
        x = np.linspace(-1.0, 5.5, 1000)
        plt.plot(x, fpoly(x))
        plt.plot([x0, x1, x2], [0.0] * 3, 'ro')
        plt.grid(True)
        plt.show()

    def test_3_generate_newton_fractal1(self):
        img = helper_function_generate_newton_fractal(fractal_functions[0:])
        self.assertTrue(img.all)

    def test_4_generate_newton_fractal2(self):
        img = helper_function_generate_newton_fractal(fractal_functions[1:])
        self.assertTrue(img.all)

    def test_5_generate_newton_fractal3(self):
        img = helper_function_generate_newton_fractal(fractal_functions[2:])
        self.assertTrue(img.all)

    def test_6_generate_newton_fractal4(self):
        img = helper_function_generate_newton_fractal(fractal_functions[3:])
        self.assertTrue(img.all)


if __name__ == '__main__':
    unittest.main()

