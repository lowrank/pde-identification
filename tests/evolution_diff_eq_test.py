import time
import unittest

from sympy import *
from sympy.abc import *

from models.evolution_diff_eq import EvolutionDiffEq

import numpy as np


class EvolutionDiffEqTest(unittest.TestCase):

    def test_1st_derivatives(self):
        model = EvolutionDiffEq(['D[u]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * x)), True)
        self.assertEqual(simplify(Eq(model.eval(sin(x)), cos(x))), True)

    def test_2nd_derivatives(self):
        model = EvolutionDiffEq(['D[D[u]]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2)), True)  # add assertion here
        self.assertEqual(simplify(Eq(model.eval(x ** 6 + x ** 4), 30 * x ** 4 + 12 * x ** 2)),
                         True)  # add assertion here

    def test_multiplication(self):
        model = EvolutionDiffEq(['D[D[u]] * u'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * (x ** 2 + 1))), True)  # add assertion here
        self.assertEqual(simplify(Eq(model.eval(sin(2 * x)), -4 * (sin(2 * x)) ** 2)), True)  # add assertion here

    def test_trig(self):
        model = EvolutionDiffEq(['D[u] * Sin[u]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * x * sin(x ** 2 + 1))), True)
        self.assertEqual(simplify(Eq(model.eval(pi * x), pi * sin(pi * x))), True)

    def test_transport_solver(self):
        dt, t_end = 0.001, 1
        t_span = np.arange(dt, t_end + dt, dt)
        x_axis = np.linspace(0, 2 * np.pi, 513)[0:-1]
        y0 = np.sin(x_axis)

        transport_model = EvolutionDiffEq(['D[u]'])
        transport_model.wrap(label="transport")

        sol = transport_model.solve(y0, t_span, rtol=1e-12, atol=1e-12)
        error = np.mean(np.abs(np.sin(x_axis + t_span[499]) - sol[500, :]))
        self.assertAlmostEqual(error, 0)

    def test_diffusion_solver(self):
        dt, t_end = 0.001, 1
        t_span = np.arange(dt, t_end + dt, dt)
        x_axis = np.linspace(0, 2 * np.pi, 513)[0:-1]
        y0 = np.sin(x_axis)

        diffusion_model = EvolutionDiffEq(['D[D[u]]'])
        diffusion_model.wrap(label="diffusion")
        sol = diffusion_model.solve(y0, t_span, rtol=1e-12, atol=1e-12)
        error = np.mean(np.abs(np.sin(x_axis) * np.exp(-t_span[499]) - sol[500, :]))
        self.assertAlmostEqual(error, 0)


if __name__ == '__main__':
    unittest.main()
