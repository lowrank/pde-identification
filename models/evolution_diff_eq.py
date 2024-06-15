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

    $$ \\partial_t u = L (u, v).   \\partial_t v = R (u, v). $$

    L, R are real-valued. The domain is fixed as [0, 2 * pi] and the derivatives are evaluated by fast Fourier transforms.

    The function $Lu$ is supplied in mathematica form. Examples are:

    1. 'D[u]' for first derivative
    2. 'D[D[u]]' for second derivative
    3. 'D[u]*u' for a product of u and its derivatives.
    4. 'D[u]*Sin[u]' for a product of sin(u) and its derivatives.

    Then ``sympy`` will parse the string into a Function.
    """

    def __init__(self, func_str_real, func_str_complex=None):
        self.model_lib = None
        self.func_str_real = func_str_real
        self.func_str_complex = func_str_complex

        if func_str_complex is None:
            self.complex = False
        else:
            self.complex = True

        self.base_repr_real = parse_mathematica(self.func_str_real)
        if self.complex:
            self.base_repr_complex = parse_mathematica(self.func_str_complex)

        self.math_repr = self._parse()

        self.model_path = os.path.dirname(os.path.abspath(__file__)) + '/.tmp/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _parse(self):
        """
        Returns:
            sympy expression representing the function without replacement.
        """
        if self.complex:
            return (self.base_repr_real.replace(Function('D'), lambda f_: Derivative(f_, x)),
                    self.base_repr_complex.replace(Function('D'), lambda f_: Derivative(f_, x)))
        else:
            return self.base_repr_real.replace(Function('D'), lambda f_: Derivative(f_, x))

    def eval(self, func_u: Function, func_v: Function = None) -> tuple[Any, Any] | Any:
        """
        evaluate a specific function defined in variable 'x'.

        Args:
            func_u: function for u variable
            func_v: function for v variable


        Returns:
            sympy expression explicitly evaluated.

        """
        if self.complex:
            return (self.math_repr[0].subs(u, func_u).subs(v, func_v).doit(),
                    self.math_repr[1].subs(u, func_u).subs(v, func_v).doit())
        else:
            return self.math_repr.subs(u, func_u).doit()

    def _dirty_codegen(self, filename="model.cpp"):
        tmp_file = self.model_path + filename
        if self.complex:
            [(_, c_code), _] = \
                codegen([('model_real', self.base_repr_real), ('model_complex', self.base_repr_complex)], 'C99',
                        'extra', header=False, empty=False)

            header = r'#include "../../src/extra.h" ' + '\n'
            source_real = ('extern "C"' + '\n' +
                           'void model_real(double* a, double* b, double* c, long n)' +
                           '{Vector u(n, true, a); Vector v(n, true, b);')
            source_real += \
                ''.join(c_code.replace('double', 'Vector').split('\n')[3:5])

            source_real += r'memcpy(c, model_real_result._data, n * sizeof(scalar_t));}' + '\n'

            source_complex = ('extern "C"' + '\n' +
                              'void model_complex(double* a, double* b, double* c, long n)' +
                              '{Vector u(n, true, a); Vector v(n, true, b);')
            source_complex += \
                ''.join(c_code.replace('double', 'Vector').split('\n')[8:10])

            source_complex += r'memcpy(c, model_complex_result._data, n * sizeof(scalar_t));}'

            source = header + source_real + source_complex

            with open(tmp_file, "w") as source_file:
                source_file.write(source)
                source_file.close()
        else:
            [(_, c_code), _] = \
                codegen(('model', self.base_repr_real), 'C99', 'extra', header=False, empty=False)

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

        if self.complex:
            getattr(self.model_lib, 'model_real').restype = None
            getattr(self.model_lib, 'model_real').argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
                np.ctypeslib.c_intp]

            getattr(self.model_lib, 'model_complex').restype = None
            getattr(self.model_lib, 'model_complex').argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
                np.ctypeslib.c_intp]

        else:
            getattr(self.model_lib, 'model').restype = None
            getattr(self.model_lib, 'model').argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED']),
                np.ctypeslib.ndpointer(dtype=np.float64, flags=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
                np.ctypeslib.c_intp]

    def compute(self, data):
        if self.complex:
            data_length = len(data)
            data_real = data[0:data_length // 2]
            data_complex = data[data_length // 2::]
            data_real = np.asanyarray(data_real)
            data_real = np.require(data_real, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
            data_complex = np.asanyarray(data_complex)
            data_complex = np.require(data_complex, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
            res_real = np.empty_like(data_real)
            res_complex = np.empty_like(data_complex)
            res_real = np.require(res_real, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
            res_complex = np.require(res_complex, dtype=np.float64,
                                     requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
            self.model_lib.model_real(data_real, data_complex, res_real, data_real.size)
            self.model_lib.model_complex(data_real, data_complex, res_complex, data_complex.size)
            return np.concatenate((res_real, res_complex))
        else:
            data = np.asanyarray(data)
            data = np.require(data, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED'])
            res = np.empty_like(data)
            res = np.require(res, dtype=np.float64, requirements=['C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
            self.model_lib.model(data, res, data.size)
            return res

    def solve(self, initial, t_span, rtol=1e-12, atol=1e-12):
        return odeint(lambda _t, _y: self.compute(_y), initial, t_span, tfirst=True, rtol=rtol, atol=atol)

    def __repr__(self):
        if self.complex:
            return "$$ \\partial_t u = %s $$\n$$ \\partial_t v = %s $$" % (
                latex(self.math_repr[0]), latex(self.math_repr[1]))
        else:
            return "$$ \\partial_t u = %s $$" % latex(self.math_repr)
