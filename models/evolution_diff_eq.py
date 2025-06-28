import os
import numpy as np
from scipy.integrate import odeint
from sympy import Function, Derivative, latex
from sympy.abc import *
from sympy.parsing.mathematica import parse_mathematica
from sympy.utilities.codegen import codegen
import jinja2
import subprocess


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

    def __init__(self, funcs_list):
        """

        Args:
            funcs_list: A list of expression on RHS
        """
        self.model_lib = None
        if len(funcs_list) == 2:
            self.func_str_real = funcs_list[0]
            self.func_str_complex = funcs_list[1]
            self.complex = True
        else:
            self.func_str_real = funcs_list[0]
            self.complex = False

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

    def eval(self, func_u: Function, func_v: Function = None):
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

    def _codegen(self, filename="model.cpp"):
        tmp_file = self.model_path + filename
        template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(os.path.abspath(__file__)))
        template_env = jinja2.Environment(loader=template_loader)

        if self.complex:
            [(_, c_code), _] = \
                codegen([('model_real', self.base_repr_real), ('model_complex', self.base_repr_complex)], 'C99',
                        'extra', header=False, empty=False)
            template = template_env.get_template('template_complex.cpp')
            source = template.render(
                model_real_code=''.join(c_code.replace('double', 'Vector').split('\n')[3:5]),
                model_complex_code=''.join(c_code.replace('double', 'Vector').split('\n')[8:10])
            )
        else:
            [(_, c_code), _] = \
                codegen(('model', self.base_repr_real), 'C99', 'extra', header=False, empty=False)
            template = template_env.get_template('template_real.cpp')
            source = template.render(
                model_code=''.join(c_code.replace('double', 'Vector').split('\n')[3:5])
            )

        with open(tmp_file, "w") as source_file:
            source_file.write(source)

    def _compile(self, filename="model.cpp", libname="model.so"):
        tmp_file = self.model_path + filename
        tmp_lib = self.model_path + libname
        c_flags = '-shared -Ofast -march=native -std=gnu++11 -ffast-math'.split(' ')
        l_flags = '-lm -lfftw3'.split(' ')

        # Compile the code with g++.
        subprocess.call(["g++", tmp_file] + c_flags + ["-o", tmp_lib] + l_flags)

    def wrap(self, label="model"):
        self._codegen(label + ".cpp")
        self._compile(label + ".cpp", label + ".so")

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
