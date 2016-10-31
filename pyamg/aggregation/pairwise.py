"""Support for aggregation-based AMG"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

import pdb 
from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr,\
    SparseEfficiencyWarning

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import relaxation_as_linear_operator,\
    eliminate_diag_dom_nodes, blocksize,\
    levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates, \
    mat_mat_complexity, unpack_arg
from .aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation, notay_pairwise, weighted_matching
from .tentative import fit_candidates
from .smooth import jacobi_prolongation_smoother,\
    richardson_prolongation_smoother, energy_prolongation_smoother

__all__ = ['pairwise_solver']


def pairwise_solver(A, B=None, BH=None,
                    symmetry='hermitian',
                    aggregate='standard',
                    smooth=('jacobi', {'omega': 4.0/3.0}),
                    presmoother=('block_gauss_seidel',
                                 {'sweep': 'symmetric'}),
                    postsmoother=('block_gauss_seidel',
                                  {'sweep': 'symmetric'}),
                    improve_candidates=[('block_gauss_seidel',
                                        {'sweep': 'symmetric',
                                         'iterations': 4}), None],
                    max_levels = 10, max_coarse = 10,
                    keep=False, **kwargs):
    """
    Create a multilevel solver using Pairwise Aggregation

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    B : {None, array_like}
        Right near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    BH : {None, array_like}
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    symmetry : {string}
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        'nonsymmetric' i.e. nonsymmetric in a hermitian sense
        Note, in the strictly real case, symmetric and hermitian are the same
        Note, this flag does not denote definiteness of the operator.
    aggregate : {list} : default ['standard', 'lloyd', 'naive', 'pairwise',
                ('predefined', {'AggOp' : csr_matrix})]
        Method used to aggregate nodes.  See notes below for varying this
        parameter on a per level basis.  Also, see notes below for using a
        predefined aggregation on each level. Method-specific parameters may be
        passed in using a tuple, e.g. aggregate=('pairwise',{'num_matchings': 2 })
    smooth : {list} : default ['jacobi', 'richardson', 'energy', None]
        Method used to smooth the tentative prolongator.  Method-specific
        parameters may be passed in using a tuple, e.g.  smooth=
        ('jacobi',{'filter' : True }).  See notes below for varying this
        parameter on a per level basis.
    presmoother : {tuple, string, list} : default ('block_gauss_seidel',
                  {'sweep':'symmetric'})
        Defines the presmoother for the multilevel cycling.  The default block
        Gauss-Seidel option defaults to point-wise Gauss-Seidel, if the matrix
        is CSR or is a BSR matrix with blocksize of 1.  See notes below for
        varying this parameter on a per level basis.
    postsmoother : {tuple, string, list}
        Same as presmoother, except defines the postsmoother.
    improve_candidates : {tuple, string, list} : default
                        [('block_gauss_seidel',
                         {'sweep': 'symmetric', 'iterations': 4}), None]
        The ith entry defines the method used to improve the candidates B on
        level i.  If the list is shorter than max_levels, then the last entry
        will define the method for all levels lower.  If tuple or string, then
        this single relaxation descriptor defines improve_candidates on all
        levels.
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
    max_levels : {integer} : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : {integer} : default 500
        Maximum number of variables permitted on the coarse grid.
    keep : {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C),
        tentative prolongation (T), and aggregation (AggOp) are kept.

    Other Parameters
    ----------------
    cycle_type : ['V','W','F']
        Structrure of multigrid cycle
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    setup_complexity : bool
        For a detailed, more accurate setup complexity, pass in 
        'setup_complexity' = True. This will slow down performance, but
        increase accuracy of complexity count. 

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    See Also
    --------
    multilevel_solver, classical.ruge_stuben_solver,
    aggregation.smoothed_aggregation_solver

    Notes
    -----



    References
    ----------



    """

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix or '
                            'bsr_matrix, or be convertible to csr_matrix')

    A = A.asfptype()

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and\
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or '
                         'hermitian\' for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Right near nullspace candidates use constant for each variable as default
    if B is None:
        B = np.kron(np.ones((int(A.shape[0]/blocksize(A)), 1), dtype=A.dtype),
                    np.eye(blocksize(A)))
    else:
        B = np.asarray(B, dtype=A.dtype)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The near null-space modes B have incorrect \
                              dimensions for matrix A')
        if B.shape[1] < blocksize(A):
            warn('Having less target vectors, B.shape[1], than \
                  blocksize of A can degrade convergence factors.')

    # Left near nullspace candidates
    if A.symmetry == 'nonsymmetric':
        if BH is None:
            BH = B.copy()
        else:
            BH = np.asarray(BH, dtype=A.dtype)
            if len(BH.shape) == 1:
                BH = BH.reshape(-1, 1)
            if BH.shape[1] != B.shape[1]:
                raise ValueError('The number of left and right near \
                                  null-space modes B and BH, must be equal')
            if BH.shape[0] != A.shape[0]:
                raise ValueError('The near null-space modes BH have \
                                  incorrect dimensions for matrix A')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    improve_candidates =\
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(multilevel_solver.level())
    levels[-1].A = A          # matrix

    # Append near nullspace candidates
    levels[-1].B = B          # right candidates
    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH    # left candidates

    while len(levels) < max_levels and\
            int(levels[-1].A.shape[0]/blocksize(levels[-1].A)) > max_coarse:
        extend_hierarchy(levels, strength, aggregate, smooth,
                         improve_candidates, diagonal_dominance, keep)

    # Construct and return multilevel hierarchy
    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def extend_hierarchy(levels, strength, aggregate, smooth, improve_candidates,
                     diagonal_dominance=False, keep=True):
    """Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.
    """

    A = levels[-1].A
    B = levels[-1].B
    if A.symmetry == "nonsymmetric":
        AH = A.H.asformat(A.format)
        BH = levels[-1].BH

    # Compute tentative interpolation operator T. Only fits one target
    # per aggregate - if more are provided, they are fit in fit_candidates().
    fn, kwargs = unpack_arg(aggregate[len(levels)-1])
    if fn == 'notay':
        AggOp = notay_pairwise(A, B=B, **kwargs)
    elif fn == 'matching':
        AggOp = weighted_matching(A, B=B, improve_candidates=
                                  improve_candidates[len(levels)-1],
                                  **kwargs)
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    levels[-1].complexity['aggregation'] = kwargs['cost'][0]

    # Improve near nullspace candidates by relaxing on A B = 0
    temp_cost = [0.0]
    fn, kwargs = unpack_arg(improve_candidates[len(levels)-1], cost=False)
    if fn is not None:
        b = np.zeros((A.shape[0], 1), dtype=A.dtype)
        B = relaxation_as_linear_operator((fn, kwargs), A, b, temp_cost) * B
        levels[-1].B = B
        if A.symmetry == "nonsymmetric":
            BH = relaxation_as_linear_operator((fn, kwargs), AH, b, temp_cost) * BH
            levels[-1].BH = BH

    levels[-1].complexity['candidates'] = temp_cost[0] * B.shape[1]

    # Compute the tentative prolongator, T, which is a tentative interpolation
    # matrix from the coarse-grid to the fine-grid.  T exactly interpolates
    # B_fine = T B_coarse. Orthogonalization complexity ~ 2nk^2, k=B.shape[1].
    # Note, if only one candidate vector is provided, T has already been
    # constructed in the pairwise aggregation process. 
    temp_cost=[0.0]
    if B.shape[1] > 1:  
        T, B = fit_candidates(AggOp, B, cost=temp_cost)
    else: 
        T = AggOp
        B = T.T*B

    if A.symmetry == "nonsymmetric":
        TH, BH = fit_candidates(AggOp, BH, cost=temp_cost)

    levels[-1].complexity['tentative'] = (temp_cost[0] + n) / float(A.nnz)

    # Smooth the tentative prolongator, so that it's accuracy is greatly
    # improved for algebraically smooth error.
    fn, kwargs = unpack_arg(smooth[len(levels)-1])
    if fn == 'jacobi':
        P = jacobi_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn == 'richardson':
        P = richardson_prolongation_smoother(A, T, **kwargs)
    elif fn == 'energy':
        P = energy_prolongation_smoother(A, T, C, B, None, (False, {}),
                                         **kwargs)
    elif fn is None:
        P = T
    else:
        raise ValueError('unrecognized prolongation smoother method %s' %
                         str(fn))

    levels[-1].complexity['smooth_P'] = kwargs['cost'][0]

    # Compute the restriction matrix, R, which interpolates from the fine-grid
    # to the coarse-grid.  If A is nonsymmetric, then R must be constructed
    # based on A.H.  Otherwise R = P.H or P.T.
    symmetry = A.symmetry
    if symmetry == 'hermitian':
        R = P.H
    elif symmetry == 'symmetric':
        R = P.T
    elif symmetry == 'nonsymmetric':
        fn, kwargs = unpack_arg(smooth[len(levels)-1])
        if fn == 'jacobi':
            R = jacobi_prolongation_smoother(AH, TH, C, BH, **kwargs).H
        elif fn == 'richardson':
            R = richardson_prolongation_smoother(AH, TH, **kwargs).H
        elif fn == 'energy':
            R = energy_prolongation_smoother(AH, TH, C, BH, None, (False, {}),
                                             **kwargs)
            R = R.H
        elif fn is None:
            R = T.H
        else:
            raise ValueError('unrecognized prolongation smoother method %s' %
                             str(fn))
        levels[-1].complexity['smooth_R'] = kwargs['cost'][0]

    if keep:
        levels[-1].C = C            # strength of connection matrix
        levels[-1].AggOp = AggOp    # aggregation operator
        levels[-1].T = T            # tentative prolongator

    levels[-1].P = P  # smoothed prolongator
    levels[-1].R = R  # restriction operator

    # Form coarse grid operator, get complexity
    levels[-1].complexity['RAP'] = mat_mat_complexity(R,A) / float(A.nnz)
    RA = R * A
    levels[-1].complexity['RAP'] += mat_mat_complexity(RA,P) / float(A.nnz)
    A = RA * P      # Galerkin operator, Ac = RAP
    A.symmetry = symmetry

    levels.append(multilevel_solver.level())
    levels[-1].A = A
    levels[-1].B = B           # right near nullspace candidates

    if A.symmetry == "nonsymmetric":
        levels[-1].BH = BH     # left near nullspace candidates
