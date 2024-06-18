import unittest

import numpy as np
from scipy.interpolate import BSpline

from utilities.function_repr import *


def generate_bsp_basis(grid_points: np.ndarray,
                       interval_bounds: tuple[float, float],
                       grid_number_for_bases: int,
                       order: int,
                       der_order: int = 0
                       ):
    """
    Generate B spline basis functions for one dimension interval specified by interval_bounds.
    The interval [lower, upper] will be uniformly divided into N (grid_number_for_bases) subintervals.
    Over each of these subintervals, we set up a Bspline basis element of order d (order).
    Then the rest of the basis functions will be added to form a complete basis.
    Args:
        grid_points (np.ndarray):
            Grid points where bases are evaluated
        interval_bounds (tuple[float, float]):
            Specify the lower and upper bounds for the interval
        grid_number_for_bases (int):
            The number of evenly spaced grid points for setting up the bases
        order (int):
            The order of the B-spline basis polynomials.
        der_order (int):
                    The order of derivatives
    """
    grid_width = (interval_bounds[1] - interval_bounds[0]) / grid_number_for_bases
    knot_width = grid_width / (order + 1)
    knot_points = interval_bounds[0] + np.arange(-order, (grid_number_for_bases + 1) * (order + 1)) * knot_width
    knot_points_number = len(knot_points)
    basis_number = knot_points_number - order - 1
    basis_mat = np.zeros((len(grid_points), basis_number))
    for n in range(basis_number):
        local_knots = knot_points[n:(n + order + 2)]
        bs_elem = BSpline.basis_element(local_knots)
        bs_elem = bs_elem.derivative(der_order)
        non_support = (grid_points > local_knots[-1]) | (grid_points < local_knots[0])
        b = bs_elem(grid_points)
        b[non_support] = 0.0
        basis_mat[:, n] = b
    return basis_mat


class FuncReprTest(unittest.TestCase):
    def test_design_matrix(self):
        b_spline_func = FunctionRepr(basis_type='b', dim=1)

        x = np.random.rand(500) * 2 * np.pi
        grid_num, k = 10, 3
        t = np.arange(-k, (grid_num + 1) * (k + 1)) * (2 * np.pi) / grid_num / (k + 1)

        for i in range(3):
            m_matrix = generate_bsp_basis(x, (0, 2 * np.pi), grid_num, k, i)
            n_matrix = b_spline_func.b_construct_design_matrix(x, t, k, i, False)
            self.assertEqual(True, np.allclose(m_matrix, n_matrix.todense(), rtol=1e-14))  # add assertion here

    def test_projection(self):
        x_ = np.linspace(0, 2 * np.pi, 33)[0:-1]
        x_samples = np.linspace(0, 2 * np.pi, 513)[0:-1]
        y_ = np.sin(3 * x_)
        y_samples = np.sin(3 * x_samples)
        k = 4
        grid_num = 11
        # (k+1) * grid_num + 2 * k + 1
        t = np.arange(-k, (grid_num + 1) * (k + 1)) * (2 * np.pi) / grid_num / (k + 1)

        func_space = FunctionRepr('b', 1)

        sample_matrix = func_space.b_construct_design_matrix(x_samples, t, k, 0, False, True)
        c = func_space.b_solve(x_, y_, t, k, True, True)
        res = sample_matrix @ c
        self.assertEqual(True, np.allclose(res, y_samples, atol=3e-3))

        c = func_space.b_solve(x_, y_, t, k, True, False)
        res = sample_matrix @ c
        self.assertEqual(False, np.allclose(res, y_samples, atol=3e-2))


if __name__ == '__main__':
    unittest.main()
