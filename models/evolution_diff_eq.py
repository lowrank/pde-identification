import os

import numpy as np
from scipy.integrate import odeint
from sympy import Function, Derivative, latex
from sympy.abc import *
from sympy.parsing.mathematica import parse_mathematica
from sympy.utilities.codegen import codegen


class EvolutionDiffEq:
    """
    Construct the equation with symbolic representation and initial boundary condition (periodic).

    $$ \\partial_t u = L u. $$

    The domain is fixed as [0, 2 * pi] and the derivatives are evaluated by fast Fourier transforms.

    The function $Lu$ is supplied in mathematica form. Examples are:

    1. 'D[u]' for first derivative
    2. 'D[D[u]]' for second derivative
    3. 'D[u]*u' for a product of u and its derivatives.
    4. 'D[u]*Sin[u]' for a product of sin(u) and its derivatives.

    Then ``sympy`` will parse the string into a Function.
    """

    def __init__(self, func_str):
        self.model_lib = None
        self.func_str = func_str
        self.base_repr = parse_mathematica(self.func_str)
        self.math_repr = self._parse()

        self.model_path = os.path.dirname(os.path.abspath(__file__)) + '/.tmp/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _parse(self) -> object:
        """
        Returns:
            sympy expression representing the function without replacement.
        """
        return self.base_repr.replace(Function('D'), lambda f_: Derivative(f_, x))

    def eval(self, func: Function) -> Function:
        """
        Args:
            func: evaluate a specific function defined in variable 'x'.

        Returns:
            sympy expression explicitly evaluated.

        """
        return self.math_repr.subs(u, func).doit()

    def _dirty_codegen(self, filename="model.cpp"):
        tmp_file = self.model_path + filename
        [(_, c_code), _] = \
            codegen(('model', self.base_repr), 'C99', 'extra', header=False, empty=False)

        header = r'#include "../../src/extra.h" ' + '\n'
        source = 'extern "C"' + '\n' + 'void model(double* a, double* b, long n) {Vector u(n, true, a);'
        source += \
            ''.join(c_code.replace('double', 'Vector').split('\n')[3:5])

        source += r'memcpy(b, model_result._data, n * sizeof(scalar_t));}'

        source = header + source

        with open(tmp_file, "w") as source_file:
            source_file.write(source)
            source_file.close()

    def _dirty_compile(self, filename="model.cpp", libname="model.so"):
        tmp_file = self.model_path + filename
        tmp_lib = self.model_path + libname
        c_flags = '-shared -Ofast -march=native -std=gnu++11 -ffast-math'.split(' ')
        l_flags = '-lm -lfftw3'.split(' ')

        # Compile the code with g++.
        import subprocess
        # use subprocess.call to ensure the compiling is finished before running the rest.
        subprocess.call(["g++", tmp_file] + c_flags + ["-o", tmp_lib] + l_flags)

    def wrap(self, label="model"):
        self._dirty_codegen(label + ".cpp")
        self._dirty_compile(label + ".cpp", label + ".so")

        self.model_lib = np.ctypeslib.load_library(label, self.model_path)
        getattr(self.model_lib, 'model').restype = None
        getattr(self.model_lib, 'model').argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
            np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
            np.ctypeslib.c_intp]

    def compute(self, data):
        data = np.asanyarray(data)
        data = np.require(data, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
        res = np.empty_like(data)
        res = np.require(res, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self.model_lib.model(data, res, data.size)
        return res

    def solve(self, initial, t_span, rtol=1e-12, atol=1e-12):
        return odeint(lambda _t, _y: self.compute(_y), initial, t_span, tfirst=True, rtol=rtol, atol=atol)

    def __repr__(self):
        return "$$ \\partial_t u = %s $$" % latex(self.math_repr)
