import numpy as np


####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    zero = np.float64(0.0)

    #  set meaningful minimal interval size if not given as parameter, e.g. 10 * eps

    if (np.isclose(ival_size, np.multiply(np.finfo(np.float64).eps, 2))):
        print("set meaningful range first")

    if (np.less(ival_size, zero)):
        ival_size = np.multiply(10, np.finfo(np.float64).eps)


    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    xplus = np.float64(0.0)
    xmin = np.float64(0.0)

    if (fl > 0 and fr < 0):
        xplus = lival
        xmin = rival
    elif (fr > 0 and fl < 0):
        xplus = rival
        xmin = lival
    else:
        print("Problem with checks for xplus and xmin")

    n_iterations = 0
    #  loop until final interval is found, stop if max iterations are reached

    #  calculate final approximation to root
    root = np.float64(0.0)
    x = np.float64(0.0)

    for i in range(0, n_iters_max):
        if not (np.isclose(np.absolute(np.subtract(xmin,xplus)), ival_size)):
            x = np.divide(np.add(xmin, xplus),2)
            if (np.isclose(f(x),np.finfo(np.float64).eps)):
                root = x
            elif (np.less(f(x), zero)):
                xmin = x
            elif (np.greater(f(x), zero)):
                xplus = x


    if (np.isclose(root, zero)):
        root = xmin



    return root


def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    #  chose meaningful convergence criterion eps, e.g 10 * eps

    bound = np.multiply(np.finfo(np.float64).eps, 10)

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    #  loop until convergence criterion eps is met
    for i in range(0, n_iters_max):

        n_iterations = n_iterations + 1

    #  return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if (np.isclose(np.absolute(dfc),np.finfo(np.float64).eps) or np.absolute(root) > 1e5):
            return (root, n_iters_max+1)

    #  update root value and function/dfunction values
        rtemp = np.copy(root)
        root = np.subtract(root, np.divide(fc, dfc))
        fc = f(root)
        dfc = df(root)

        if (np.less(np.absolute(np.subtract(root, rtemp)), bound)):
            break

    #  avoid infinite loops and return (root, n_iters_max+1)

    if (n_iterations == n_iters_max):
        return root, n_iters_max+1

    return root, n_iterations

####################################################################################################
# Exercise 2: Newton Fractal


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it.
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)

    #  iterate over sampling grid
    for i in range(0, sampling[0].size):
        for j in range(0, sampling[0].size):
            #  run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            root, num_it = find_root_newton(f, df, sampling[i][j], n_iters_max)
            #  determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            index = 0
            for z in range(0, roots.size):
                if (np.isclose(root, roots[z])):
                    index = z
                    break

            #  write the index and the number of needed iterations to the result
            result[i, j] = np.array([index, num_it])

    return result


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
