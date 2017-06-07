"""Ideal restriction AMG"""
from __future__ import absolute_import

__docformat__ = "restructuredtext en"

from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning, block_diag
import numpy as np
from copy import deepcopy

from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    distance_strength_of_connection, energy_based_strength_of_connection, \
    algebraic_distance, affinity_distance
from pyamg.util.utils import mat_mat_complexity, unpack_arg, extract_diagonal_blocks, \
    filter_matrix_rows
from pyamg.classical.interpolate import direct_interpolation, standard_interpolation, \
     trivial_interpolation, injection_interpolation, approximate_ideal_restriction, \
     neumann_ideal_restriction, neumann_ideal_interpolation
from pyamg.classical.split import *
from pyamg.classical.cr import CR

__all__ = ['AMGir_solver']

def AMGir_solver(A,
                 strength=('classical', {'theta': 0.3 ,'norm': 'min'}),
                 CF='RS',
                 interp='one_point',
                 restrict='neumann',
                 presmoother=None,
                 postsmoother=('FC_jacobi', {'omega': 1.0, 'iterations': 1,
                                'withrho': False,  'F_iterations': 2,
                                'C_iterations': 0} ),
                 filter_operator=None,
                 coarse_grid_P=None, 
                 coarse_grid_R=None, 
                 max_levels=20, max_coarse=20,
                 keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG)

    Parameters
    ----------
    A : csr_matrix
        Square nonsymmetric matrix in CSR format
    strength : ['symmetric', 'classical', 'evolution', 'distance',
                'algebraic_distance','affinity', 'energy_based', None]
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta' : 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : {string} : default 'RS'
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    interp : {string} : default 'inject'
        Options include 'direct', 'standard', 'inject' and 'trivial'.
    restrict : {string} : default 'neumann'
        Options include 'air' for approximate ideal
        restriction.
    presmoother : {string or dict} : default None
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : {string or dict} : default F-Jacobi
        Postsmoothing method with the same usage as presmoother
    filter_operator : (bool, tol) : default None
        Remove small entries in operators on each level if True. Entries are
        considered "small" if |a_ij| < tol |a_ii|.
    coarse_grid_P : {string} : default None
        Option to specify a different construction of P used in computing RAP
        vs. for interpolation in an actual solve.
    max_levels: {integer} : default 20
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer} : default 20
        Maximum number of variables permitted on the coarse grid.
    keep: {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C) and
        tentative prolongation (T) are kept.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Other Parameters
    ----------------
    coarse_solver : ['splu', 'lu', 'cholesky, 'pinv', 'gauss_seidel', ... ]
        Solver used at the coarsest level of the MG hierarchy.
            Optionally, may be a tuple (fn, args), where fn is a string such as
        ['splu', 'lu', ...] or a callable function, and args is a dictionary of
        arguments to be passed to fn.
    setup_complexity : bool
        For a detailed, more accurate setup complexity, pass in 
        'setup_complexity' = True. This will slow down performance, but
        increase accuracy of complexity count. 

    Notes
    -----




    References
    ----------
    .. [1] 

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """

    if ('setup_complexity' in kwargs):
        if kwargs['setup_complexity'] == True:
            mat_mat_complexity.__detailed__ = True
        del kwargs['setup_complexity']

    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels = [multilevel_solver.level()]
    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        bottom = extend_hierarchy(levels, strength, CF, interp, restrict, filter_operator,
                                  coarse_grid_P, coarse_grid_R, keep)
        if bottom:
            break

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, strength, CF, interp, restrict, filter_operator,
                     coarse_grid_P, coarse_grid_R, keep):
    """ helper function for local methods """

    # Filter operator. Need to keep original matrix on fineest level for
    # computing residuals
    if (filter_operator is not None) and (filter_operator[1] != 0): 
        if len(levels) == 1:
            A = deepcopy(levels[-1].A)
        else:
            A = levels[-1].A
        filter_matrix_rows(A, filter_operator[1], diagonal=True, lump=filter_operator[0])
    else:
        A = levels[-1].A

    # Check if matrix was filtered to be diagonal --> coarsest grid
    if A.nnz == A.shape[0]:
        return 1

    # Zero initial complexities for strength, splitting and interpolation
    levels[-1].complexity['CF'] = 0.0
    levels[-1].complexity['strength'] = 0.0
    levels[-1].complexity['interpolate'] = 0.0

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        C = evolution_strength_of_connection(A, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        C = A
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))
    levels[-1].complexity['strength'] += kwargs['cost'][0] * A.nnz / float(A.nnz)

    # Generate the C/F splitting
    fn, kwargs = unpack_arg(CF)
    if fn == 'RS':
        splitting = RS(C, **kwargs)
    elif fn == 'PMIS':
        splitting = PMIS(C, **kwargs)
    elif fn == 'PMISc':
        splitting = PMISc(C, **kwargs)
    elif fn == 'CLJP':
        splitting = CLJP(C, **kwargs)
    elif fn == 'CLJPc':
        splitting = CLJPc(C, **kwargs)
    elif fn == 'CR':
        splitting = CR(C, **kwargs)
    elif fn == 'weighted_matching':
        splitting, soc = weighted_matching(C, **kwargs)
        if soc is not None:
            C = soc
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)
    levels[-1].complexity['CF'] += kwargs['cost'][0] * C.nnz / float(A.nnz)

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    r_flag = False
    fn, kwargs = unpack_arg(interp)
    if fn == 'standard':
        P = standard_interpolation(A, C, splitting, **kwargs)
    elif fn == 'distance_two':
        P = distance_two_interpolation(A, C, splitting, **kwargs)
    elif fn == 'direct':
        P = direct_interpolation(A, C, splitting, **kwargs)
    elif fn == 'one_point':
        P = one_point_interpolation(A, C, splitting, **kwargs)
    elif fn == 'inject':
        P = injection_interpolation(A, splitting, **kwargs)
    elif fn == 'neumann':
        P = neumann_ideal_interpolation(A, splitting, **kwargs)
    elif fn == 'air':
	    if isspmatrix_bsr(A): 
            temp_A = bsr_matrix(A.T)
            P = approximate_ideal_restriction(temp_A, splitting, **kwargs)
            P = bsr_matrix(P.T)
        else:
            temp_A = csr_matrix(A.T)
            P = approximate_ideal_restriction(temp_A, splitting, **kwargs)
            P = csr_matrix(P.T)
    elif fn == 'restrict':
        r_flag = True
    else:
        raise ValueError('unknown interpolation method (%s)' % interp)
    levels[-1].complexity['interpolate'] += kwargs['cost'][0] * A.nnz / float(A.nnz)

    # Build restriction operator
    fn, kwargs = unpack_arg(restrict)
    if fn is None:
        R = P.T
    elif fn == 'air':
        R = approximate_ideal_restriction(A, splitting, **kwargs)
    elif fn == 'neumann':
        R = neumann_ideal_restriction(A, splitting, **kwargs)
    elif fn == 'one_point':         # Don't need A^T here
        temp_C = C.T.tocsr()
        R = one_point_interpolation(A, temp_C, splitting, **kwargs)
        if isspmatrix_bsr(A):
            R = R.T.tobsr()
        else:
            R = R.T.tocsr()
    elif fn == 'inject':            # Don't need A^T or C^T here
        R = injection_interpolation(A, splitting, **kwargs)
        if isspmatrix_bsr(A):
            R = R.T.tobsr()
        else:
            R = R.T.tocsr()
    elif fn == 'standard':
        if isspmatrix_bsr(A):
            temp_A = A.T.tobsr()
            temp_C = C.T.tobsr()
            R = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()
        else: 
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
    elif fn == 'direct':
        if isspmatrix_bsr(A):
            temp_A = A.T.tobsr()
            temp_C = C.T.tobsr()
            R = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tobsr()        
        else:
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R = R.T.tocsr()
    else:
        raise ValueError('unknown restriction method (%s)' % restrict)

    # If set P = R^T
    if r_flag:
        P = R.T

    # Optional different interpolation for RAP
    fn, kwargs = unpack_arg(coarse_grid_P)
    if fn == 'standard':
        P_temp = standard_interpolation(A, C, splitting, **kwargs)
    elif fn == 'distance_two':
        P_temp = distance_two_interpolation(A, C, splitting, **kwargs)
    elif fn == 'direct':
        P_temp = direct_interpolation(A, C, splitting, **kwargs)
    elif fn == 'one_point':
        P_temp = one_point_interpolation(A, C, splitting, **kwargs)
    elif fn == 'inject':
        P_temp = injection_interpolation(A, splitting, **kwargs)
    elif fn == 'neumann':
        P_temp = neumann_ideal_interpolation(A, splitting, **kwargs)
    elif fn == 'air':
        if isspmatrix_bsr(A): 
            temp_A = bsr_matrix(A.T)
            P_temp = approximate_ideal_restriction(temp_A, splitting, **kwargs)
            P_temp = bsr_matrix(P_temp.T)
        else:
            temp_A = csr_matrix(A.T)
            P_temp = approximate_ideal_restriction(temp_A, splitting, **kwargs)
            P_temp = csr_matrix(P_temp.T)
    else:
        P_temp = P

    # Optional different restriction for RAP
    fn, kwargs = unpack_arg(coarse_grid_R)
    if fn == 'air':
        R_temp = approximate_ideal_restriction(A, splitting, **kwargs)
    elif fn == 'neumann':
        R_temp = neumann_ideal_restriction(A, splitting, **kwargs)
    elif fn == 'one_point':         # Don't need A^T here
        temp_C = C.T.tocsr()
        R_temp = one_point_interpolation(A, temp_C, splitting, **kwargs)
        if isspmatrix_bsr(A):
            R_temp = R_temp.T.tobsr()
        else:
            R_temp = R_temp.T.tocsr()
    elif fn == 'inject':            # Don't need A^T or C^T here
        R_temp = injection_interpolation(A, splitting, **kwargs)
        if isspmatrix_bsr(A):
            R_temp = R_temp.T.tobsr()
        else:
            R_temp = R_temp.T.tocsr()
    elif fn == 'standard':
        if isspmatrix_bsr(A):
            temp_A = A.T.tobsr()
            temp_C = C.T.tobsr()
            R_temp = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R_temp = R_temp.T.tobsr()
        else: 
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R_temp = standard_interpolation(temp_A, temp_C, splitting, **kwargs)
            R_temp = R_temp.T.tocsr()
    elif fn == 'direct':
        if isspmatrix_bsr(A):
            temp_A = A.T.tobsr()
            temp_C = C.T.tobsr()
            R_temp = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R_temp = R_temp.T.tobsr()        
        else:
            temp_A = A.T.tocsr()
            temp_C = C.T.tocsr()
            R_temp = direct_interpolation(temp_A, temp_C, splitting, **kwargs)
            R_temp = R_temp.T.tocsr()
    else:
        R_temp = R

    # Store relevant information for this level
    if keep:
        levels[-1].C = C              # strength of connection matrix

    levels[-1].P = P                  # prolongation operator
    levels[-1].R = R                  # restriction operator
    levels[-1].splitting = splitting  # C/F splitting

    # Form coarse grid operator, get complexity
    levels[-1].complexity['RAP'] = mat_mat_complexity(R_temp,A) / float(A.nnz)
    RA = R_temp * A
    levels[-1].complexity['RAP'] += mat_mat_complexity(RA,P_temp) / float(A.nnz)
    A = RA * P_temp

    # Make sure coarse-grid operator is in correct sparse format
    if (isspmatrix_csr(P) and (not isspmatrix_csr(A))):
        A = A.tocsr()
    elif (isspmatrix_bsr(P) and (not isspmatrix_bsr(A))):
        A = A.tobsr()

    levels.append(multilevel_solver.level())
    levels[-1].A = A
    return 0

