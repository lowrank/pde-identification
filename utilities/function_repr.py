import numpy as np
from scipy.sparse import csr_array

from . import design_matrix

"""
Function Representation class.
"""

# b: b-spline
# w: wavelet
# n: neural network

__supported__ = {'b': 'linear', 'w': 'linear', 'n': 'nonlinear'}


class FunctionRepr(object):
    def __init__(self, basis_type='b', dim=1):
        self.type = basis_type
        self.dim = dim
        self.linear = __supported__[basis_type] == 'linear'

    @classmethod
    def _b_construct_design_matrix(cls, eval_pts, knots, degree, derivative=0, periodic=True):
        """
        Create the design matrix for B splines.

        Args:
            eval_pts: point coordinates for evaluation
            knots: knot points coordinates
            degree: degree of B spline
            derivative: derivative order of B spline
            periodic: whether the knots form a periodic domain

        Returns:
            sparse matrix of shape (num of eval_pts, num of basis)
        """

        pass

    @classmethod
    def _b_construct_null_space(cls, matrix=None):
        """

        Args:
            matrix: rank-deficient matrix

        Returns:
            null space matrix through QR decomposition.
        """
        pass

    def _b_gen_function(self, coefficient):
        pass


def make_design_matrix(x, t, k, nu, extrapolate):
    """

    Args:
        x: evaluation points
        t: knots
        k: degree
        nu: derivative
        extrapolate: bool or "periodic"

    Returns:
        design matrix [ B^{(nu)}_i(x_k) ]_{k, i}
    """
    x = np.ascontiguousarray(x).astype(np.float64, copy=False)
    t = np.ascontiguousarray(t).astype(np.float64, copy=False)

    if extrapolate != 'periodic':
        extrapolate = bool(extrapolate)

    if k < 0:
        raise ValueError("Spline order cannot be negative.")
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError(f"Expect t to be a 1-D sorted array_like, but "
                         f"got t={t}.")
    # There are `nt - k - 1` basis elements in a BSpline built on the
    # vector of knots with length `nt`, so to have at least `k + 1` basis
    # elements we need to have at least `2 * k + 2` elements in the vector
    # of knots.
    if len(t) < 2 * k + 2:
        raise ValueError(f"Length t is not enough for k={k}.")

    if extrapolate == 'periodic':
        # With periodic extrapolation we map x to the segment
        # [t[k], t[n]].
        n = t.size - k - 1
        x = t[k] + (x - t[k]) % (t[n] - t[k])
        extrapolate = False
    elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
    ):
        # Checks from `find_interval` function
        raise ValueError(f'Out of bounds w/ x = {x}.')

    # Compute number of non-zeros of final CSR array in order to determine
    # the dtype of indices and indptr of the CSR array.

    n = x.shape[0]
    nnz = n * (k + 1)
    if nnz < np.iinfo(np.int32).max:
        int_dtype = np.int32
    else:
        int_dtype = np.int64
    # Preallocate indptr and indices
    indices = np.empty(n * (k + 1), dtype=int_dtype)
    indptr = np.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)

    # indptr is not passed to Cython as it is already fully computed
    data, indices = design_matrix._make_design_matrix(
        x, t, k, nu, extrapolate, indices
    )
    return csr_array(
        (data, indices, indptr),
        shape=(x.shape[0], t.shape[0] - k - 1)
    )
